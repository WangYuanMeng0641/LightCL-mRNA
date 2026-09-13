import re, os, sys
import math
import torch
import argparse
import random
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, hamming_loss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from tqdm import tqdm
import time
import torch.nn.functional as F
import psutil
import warnings

warnings.filterwarnings('ignore')

# ========== 智能CPU和内存管理 ==========
import multiprocessing


def get_optimal_threads():
    cpu_count = multiprocessing.cpu_count()
    physical_cores = cpu_count // 2
    safe_threads = min(physical_cores // 2, 4)
    safe_threads = max(safe_threads, 2)
    print(f"系统CPU核心数: {cpu_count} (逻辑核心)")
    print(f"估计物理核心数: {physical_cores}")
    print(f"安全线程数: {safe_threads}")
    return safe_threads


def get_memory_info():
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / (1024 ** 3):.1f} GB")
    print(f"可用内存: {memory.available / (1024 ** 3):.1f} GB")
    print(f"内存使用率: {memory.percent}%")
    return memory


def configure_cpu_threads():
    safe_threads = get_optimal_threads()
    os.environ["MKL_NUM_THREADS"] = str(safe_threads)
    os.environ["OMP_NUM_THREADS"] = str(safe_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(safe_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(safe_threads)
    torch.set_num_threads(safe_threads)
    torch.set_num_interop_threads(min(safe_threads, 2))
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
        print("✅ MKL-DNN已启用")
    print(f"✅ CPU优化完成，使用 {safe_threads} 个线程")
    return safe_threads


base_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== 三尺度CNN + 4头自注意力（kernel_sizes=[3,5,7]，输出192维） ====================
class FourScaleCNNWithMultiHeadAttention(nn.Module):
    def __init__(self, input_channels=4, num_filters=32, kernel_sizes=[3, 5, 7],
                 num_heads=4, dropout=0.15):
        super().__init__()

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        # 3个卷积层，每个输出32维
        self.convs = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 拼接后维度: 32 × 3 = 96
        self.conv_dim = num_filters * len(kernel_sizes)  # = 96

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 注意力层：输入192维（池化拼接后），输出192维
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.conv_dim * 2,  # = 192
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(self.conv_dim * 2)  # 192
        self.attn_dropout = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = self.conv_dim * 2  # = 192

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, 4, 4000]

        # 多尺度卷积
        conv_outs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            out = conv(x)
            out = bn(out)
            out = self.relu(out)
            out = self.pool(out)
            conv_outs.append(out)
        out = torch.cat(conv_outs, dim=1)  # [batch, 96, 2000]

        # 全局池化：Avg + Max → 192维
        avg_pooled = self.global_avg_pool(out)  # [batch, 96, 1]
        max_pooled = self.global_max_pool(out)  # [batch, 96, 1]
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [batch, 192, 1]
        pooled = pooled.squeeze(-1)  # [batch, 192]

        # 注意力：输入192维，输出192维
        pooled = pooled.unsqueeze(1)  # [batch, 1, 192]
        attn_out, attn_weights = self.multihead_attention(pooled, pooled, pooled)
        out = self.attn_norm(pooled + self.attn_dropout(attn_out))
        out = out.squeeze(1)  # [batch, 192]
        out = self.dropout(out)

        return out, attn_weights.mean().item() if attn_weights is not None else 0.0


# ==================== 阶段1: K-mer编码器（预训练） ====================
class KmerEncoderPretrain(nn.Module):
    """
    3-6mer编码器: 5440 → 3000 → 1000 → 256 → 9
    预训练阶段只使用BCE损失
    """

    def __init__(self, input_dim=5440, hidden1=3000, hidden2=1000, embed_dim=256, num_classes=9, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embed=False):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        if return_embed:
            return logits, embed
        return logits


class SevenMerEncoderPretrain(nn.Module):
    """
    7-mer编码器: 16384 → 8000 → 3000 → 1000 → 256 → 9
    预训练阶段只使用BCE损失
    """

    def __init__(self, input_dim=16384, hidden1=8000, hidden2=3000, hidden3=1000, embed_dim=256, num_classes=9,
                 dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embed=False):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        if return_embed:
            return logits, embed
        return logits


# ==================== 多标签监督对比损失 ====================
class MultiLabelSupervisedContrastiveLoss(nn.Module):
    """
    多标签监督对比损失
    基于标签重叠度（Jaccard相似度）来定义正负样本对
    重叠度 >= threshold 视为正样本对，< threshold 视为负样本对
    """

    def __init__(self, temperature=0.5, margin=1.0, overlap_threshold=0.4):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.overlap_threshold = overlap_threshold

    def compute_jaccard_similarity(self, labels_i, labels_j):
        intersection = (labels_i * labels_j).sum(dim=1)
        union = (labels_i + labels_j).clamp(0, 1).sum(dim=1)
        return intersection / (union + 1e-8)

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        overlap_matrix = torch.zeros(batch_size, batch_size, device=embeddings.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    overlap = self.compute_jaccard_similarity(
                        labels[i].unsqueeze(0),
                        labels[j].unsqueeze(0)
                    )
                    overlap_matrix[i, j] = overlap

        pos_mask = overlap_matrix >= self.overlap_threshold
        neg_mask = overlap_matrix < self.overlap_threshold

        loss = 0.0
        num_pos_pairs = 0
        num_neg_pairs = 0

        for i in range(batch_size):
            pos_indices = torch.where(pos_mask[i])[0]
            if len(pos_indices) > 0:
                pos_sim = sim_matrix[i, pos_indices]
                pos_loss = -torch.log(torch.exp(pos_sim).sum() + 1e-8)
                loss += pos_loss
                num_pos_pairs += len(pos_indices)

            neg_indices = torch.where(neg_mask[i])[0]
            if len(neg_indices) > 0:
                neg_sim = sim_matrix[i, neg_indices]
                neg_loss = F.relu(self.margin - neg_sim).pow(2).sum()
                loss += neg_loss
                num_neg_pairs += len(neg_indices)

        if num_pos_pairs + num_neg_pairs > 0:
            loss = loss / (num_pos_pairs + num_neg_pairs + 1e-8)

        return loss


# ==================== 主模型（三路融合 + 对比学习） ====================
class MultiLabelModel(nn.Module):
    """
    多标签分类模型（9个标签同时预测）
    - 分支1: One-hot → CNN+注意力 (kernel_sizes=[3,5,7]) → 192维
    - 分支2: K-mer(3-6) → 预训练MLP → 256维 (冻结)
    - 分支3: 7-mer → 预训练MLP → 256维 (冻结)
    - 拼接: 192 + 256 + 256 = 704维
    - 对比学习: 直接使用704维特征
    - MLP分类头: 704 → 256 → 64 → 9
    """

    def __init__(self, num_classes=9, dropout=0.15):
        super().__init__()
        self.num_classes = num_classes

        # ========== 分支1: One-hot CNN编码器（kernel_sizes=[3,5,7]，输出192维） ==========
        self.onehot_encoder = FourScaleCNNWithMultiHeadAttention(
            input_channels=4,
            num_filters=32,
            kernel_sizes=[3, 5, 7],
            num_heads=4,
            dropout=dropout
        )

        # ========== 分支2: K-mer(3-6) 预训练编码器 ==========
        self.kmer_encoder = KmerEncoderPretrain(
            input_dim=5440,
            hidden1=3000,
            hidden2=1000,
            embed_dim=256,
            num_classes=num_classes,
            dropout=dropout
        )

        # ========== 分支3: 7-mer 预训练编码器 ==========
        self.sevenmer_encoder = SevenMerEncoderPretrain(
            input_dim=16384,
            hidden1=8000,
            hidden2=3000,
            hidden3=1000,
            embed_dim=256,
            num_classes=num_classes,
            dropout=dropout
        )

        # ========== 拼接维度: 192 + 256 + 256 = 704 ==========
        self.total_dim = 192 + 256 + 256

        # 批归一化
        self.fusion_bn = nn.BatchNorm1d(self.total_dim)

        # ========== MLP分类头: 704 → 256 → 64 → 9 ==========
        self.classifier = nn.Sequential(
            nn.Linear(self.total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        print(f"MultiLabelModel 架构（三路特征融合，含预训练k-mer编码器）:")
        print(f"  - 分支1 (One-hot): CNN+注意力 (kernel_sizes=3,5,7) → 192维")
        print(f"  - 分支2 (K-mer 3-6): 预训练MLP 5440→3000→1000→256 → 256维 (冻结)")
        print(f"  - 分支3 (7-mer): 预训练MLP 16384→8000→3000→1000→256 → 256维 (冻结)")
        print(f"  - 融合: 拼接 → {self.total_dim}维")
        print(f"  - 对比学习: 直接使用 {self.total_dim}维融合特征")
        print(f"  - MLP分类头: {self.total_dim} → 256 → 64 → {num_classes}")

    def freeze_kmer_encoders(self):
        """冻结预训练的k-mer编码器"""
        for param in self.kmer_encoder.parameters():
            param.requires_grad = False
        for param in self.sevenmer_encoder.parameters():
            param.requires_grad = False
        print("✅ K-mer编码器已冻结")

    def forward(self, onehot, kmer_3_6, kmer_7, return_features=False):
        """
        onehot: [batch, 4000, 4] One-hot编码
        kmer_3_6: [batch, 5440] 3-6mer频次向量
        kmer_7: [batch, 16384] 7-mer频次向量
        return_features: 是否返回融合特征
        """
        # ========== 分支1: One-hot 特征 ==========
        onehot_feat, _ = self.onehot_encoder(onehot)  # [batch, 192]

        # ========== 分支2: K-mer(3-6) 特征 ==========
        _, kmer_feat = self.kmer_encoder(kmer_3_6, return_embed=True)  # [batch, 256]

        # ========== 分支3: 7-mer 特征 ==========
        _, sevenmer_feat = self.sevenmer_encoder(kmer_7, return_embed=True)  # [batch, 256]

        # ========== 特征拼接 ==========
        fused = torch.cat([onehot_feat, kmer_feat, sevenmer_feat], dim=1)  # [batch, 704]

        # 批归一化
        fused = self.fusion_bn(fused)

        # 分类
        logits = self.classifier(fused)  # [batch, num_classes]

        if return_features:
            return logits, fused

        return logits


# ==================== 预训练函数（含早停） ====================
def pretrain_kmer_encoder(model, train_loader, val_loader, opt, device, encoder_type='kmer'):
    """
    预训练单个k-mer编码器（只使用BCE损失）
    加入早停机制: 验证集连续 patience 轮不提升则停止
    """
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=8e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    bce_criterion = nn.BCEWithLogitsLoss()

    best_accuracy = 0.0
    patience_counter = 0
    best_epoch = 0
    model_path = os.path.join(opt.output_path, f'{encoder_type}_pretrain_best.pt')

    print(f"\n{'=' * 60}")
    print(f"预训练 {encoder_type.upper()} 编码器")
    print(f"  仅使用BCE损失函数")
    print(f"  epochs = {opt.pretrain_epochs}, patience = {opt.patience}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, opt.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        train_bar = tqdm(train_loader, desc=f'{encoder_type} PreTrain Epoch {epoch}/{opt.pretrain_epochs}')

        for onehot, kmer_3_6, kmer_7, labels, seq_ids in train_bar:
            if encoder_type == 'kmer':
                x = kmer_3_6.to(device)
            else:  # sevenmer
                x = kmer_7.to(device)
            labels = labels.to(device)

            logits = model(x)
            loss = bce_criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # 验证
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for onehot, kmer_3_6, kmer_7, labels, seq_ids in tqdm(val_loader, desc=f'{encoder_type} Validating'):
                if encoder_type == 'kmer':
                    x = kmer_3_6.to(device)
                else:
                    x = kmer_7.to(device)
                labels = labels.to(device)

                logits = model(x)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_multilabel_metrics(all_labels, all_logits)

        scheduler.step(metrics['accuracy'])

        print(f'\n{encoder_type} Epoch {epoch}/{opt.pretrain_epochs}:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Val: Accuracy={metrics["accuracy"]:.4f}, Aiming={metrics["aiming"]:.4f}, '
              f'Coverage={metrics["coverage"]:.4f}, Abs_True={metrics["absolute_true"]:.4f}, '
              f'Abs_False={metrics["absolute_false"]:.4f}')

        # ========== 早停判断 ==========
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f'  ✅ 新的最佳模型已保存! (Accuracy: {best_accuracy:.4f} @ Epoch {epoch})')
        else:
            patience_counter += 1
            print(f'  ⏳ 验证集Accuracy未提升 ({patience_counter}/{opt.patience})')
            if patience_counter >= opt.patience:
                print(f'  🛑 Early stopping at epoch {epoch}, 最佳Accuracy: {best_accuracy:.4f} @ Epoch {best_epoch}')
                break

        clear_memory()

    # 加载最佳模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ 已加载 {encoder_type} 最佳模型 (Accuracy: {best_accuracy:.4f})")
    else:
        print(f"\n⚠️ 警告: {encoder_type} 未保存任何模型，使用最后epoch的权重")

    print(f"\n{encoder_type} 预训练完成！")
    print(f"  最佳验证 Accuracy: {best_accuracy:.4f}")
    print(f"  最佳 Epoch: {best_epoch}")
    return best_accuracy


# ==================== 数据集类（支持三路特征） ====================
class MultiLabelDataset(Dataset):
    def __init__(self, seq_ids, onehot_features, kmer_3_6_features, kmer_7_features, labels_dict, num_classes=9):
        self.seq_ids = seq_ids
        self.onehot = [onehot_features[sid] for sid in seq_ids]
        self.kmer_3_6 = [kmer_3_6_features[sid] for sid in seq_ids]
        self.kmer_7 = [kmer_7_features[sid] for sid in seq_ids]
        self.labels_dict = labels_dict
        self.num_classes = num_classes

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        onehot = torch.from_numpy(self.onehot[idx]).float()
        kmer_3_6 = torch.tensor(self.kmer_3_6[idx], dtype=torch.float32)
        kmer_7 = torch.tensor(self.kmer_7[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels_dict[seq_id][:self.num_classes], dtype=torch.float32)
        return onehot, kmer_3_6, kmer_7, labels, seq_id


# ==================== 工具函数 ====================
def extract_onehot_features(sequence, target_len=4000):
    nucleotides = ['A', 'C', 'G', 'T']
    nuc_to_idx = {nuc: i for i, nuc in enumerate(nucleotides)}
    onehot = np.zeros((target_len, 4), dtype=np.float32)
    seq_len = len(sequence)
    if seq_len >= target_len:
        half = target_len // 2
        start_seq = sequence[:half]
        end_seq = sequence[-half:]
        combined_seq = start_seq + end_seq
        for i, nuc in enumerate(combined_seq):
            if i < target_len and nuc in nuc_to_idx:
                onehot[i, nuc_to_idx[nuc]] = 1.0
    else:
        for i, nuc in enumerate(sequence):
            if i < target_len and nuc in nuc_to_idx:
                onehot[i, nuc_to_idx[nuc]] = 1.0
    return onehot


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def read_nucleotide_sequences(file):
    print(f"正在读取FASTA文件: {file}")
    if not os.path.exists(file):
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) is None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in tqdm(records, desc="解析FASTA记录"):
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([header, sequence])
    print(f"成功读取 {len(fasta_sequences)} 条序列")
    return fasta_sequences


def compute_multilabel_metrics(y_true, y_logits):
    """
    计算多标签评估指标（与图模型代码完全一致）
    """
    y_prob = torch.sigmoid(torch.from_numpy(y_logits) if isinstance(y_logits, np.ndarray) else y_logits)
    y_prob = y_prob.numpy()
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = (y_prob > 0.5).astype(int)

    aiming = np.mean([
        np.sum(y_true_np[i] * y_pred[i]) / np.sum(y_pred[i])
        if np.sum(y_pred[i]) > 0 else 0
        for i in range(len(y_true_np))
    ])

    coverage = np.mean([
        np.sum(y_true_np[i] * y_pred[i]) / np.sum(y_true_np[i])
        if np.sum(y_true_np[i]) > 0 else 0
        for i in range(len(y_true_np))
    ])

    accuracy = jaccard_score(y_true_np, y_pred, average='samples')

    absolute_true = np.mean([
        np.array_equal(y_true_np[i], y_pred[i])
        for i in range(len(y_true_np))
    ])

    absolute_false = hamming_loss(y_true_np, y_pred)

    return {
        'aiming': aiming,
        'coverage': coverage,
        'accuracy': accuracy,
        'absolute_true': absolute_true,
        'absolute_false': absolute_false
    }


# ==================== 第二阶段训练函数（联合损失 + 对比学习） ====================
def train_stage2(model, train_loader, val_loader, opt, device):
    """
    第二阶段：三路融合 + 对比学习 + BCE 联合训练
    同样加入早停机制
    """
    # 只优化未冻结的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=8e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    bce_criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = MultiLabelSupervisedContrastiveLoss(
        temperature=opt.temperature,
        margin=opt.contrastive_margin,
        overlap_threshold=opt.overlap_threshold
    )

    best_accuracy = 0.0
    patience_counter = 0
    best_epoch = 0
    model_path = os.path.join(opt.output_path, 'best_stage2_model.pt')

    print(f"\n{'=' * 60}")
    print(f"第二阶段训练 - 三路融合 + 对比学习 + BCE")
    print(f"  epochs = {opt.stage2_epochs}, patience = {opt.patience}")
    print(f"  α (BCE权重) = {opt.alpha}")
    print(f"  1-α (对比损失权重) = {1 - opt.alpha}")
    print(f"  重叠度阈值 = {opt.overlap_threshold}")
    print(f"  Temperature = {opt.temperature}")
    print(f"  Margin = {opt.contrastive_margin}")
    print(f"  融合特征维度 = 704")
    print(f"  K-mer编码器: 冻结 ❄️")
    print(f"{'=' * 60}\n")

    for epoch in range(1, opt.stage2_epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        train_bar = tqdm(train_loader, desc=f'Stage2 Epoch {epoch}/{opt.stage2_epochs}')

        for onehot, kmer_3_6, kmer_7, labels, seq_ids in train_bar:
            onehot = onehot.to(device)
            kmer_3_6 = kmer_3_6.to(device)
            kmer_7 = kmer_7.to(device)
            labels = labels.to(device)

            logits, features = model(onehot, kmer_3_6, kmer_7, return_features=True)

            bce_loss = bce_criterion(logits, labels)
            contrastive_loss = contrastive_criterion(features, labels)
            loss = opt.alpha * bce_loss + (1 - opt.alpha) * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batches += 1

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bce': f'{bce_loss.item():.4f}',
                'cont': f'{contrastive_loss.item():.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_bce = total_bce_loss / num_batches
        avg_cont = total_contrastive_loss / num_batches

        # 验证
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for onehot, kmer_3_6, kmer_7, labels, seq_ids in tqdm(val_loader, desc='Validating'):
                onehot = onehot.to(device)
                kmer_3_6 = kmer_3_6.to(device)
                kmer_7 = kmer_7.to(device)
                labels = labels.to(device)

                logits = model(onehot, kmer_3_6, kmer_7)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_multilabel_metrics(all_labels, all_logits)

        scheduler.step(metrics['accuracy'])

        print(f'\nStage2 Epoch {epoch}/{opt.stage2_epochs}:')
        print(f'  Loss: total={avg_loss:.4f}, BCE={avg_bce:.4f}, Contrastive={avg_cont:.4f}')
        print(f'  Val: Accuracy={metrics["accuracy"]:.4f}, Aiming={metrics["aiming"]:.4f}, '
              f'Coverage={metrics["coverage"]:.4f}, Abs_True={metrics["absolute_true"]:.4f}, '
              f'Abs_False={metrics["absolute_false"]:.4f}')

        # ========== 早停判断 ==========
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f'  ✅ 新的最佳模型已保存! (Accuracy: {best_accuracy:.4f} @ Epoch {epoch})')
        else:
            patience_counter += 1
            print(f'  ⏳ 验证集Accuracy未提升 ({patience_counter}/{opt.patience})')
            if patience_counter >= opt.patience:
                print(f'  🛑 Early stopping at epoch {epoch}, 最佳Accuracy: {best_accuracy:.4f} @ Epoch {best_epoch}')
                break

        clear_memory()

    # 加载最佳模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ 已加载第二阶段最佳模型 (Accuracy: {best_accuracy:.4f})")
    else:
        print(f"\n⚠️ 警告: 第二阶段未保存任何模型，使用最后epoch的权重")

    print(f"\n第二阶段训练完成！")
    print(f"  最佳验证 Accuracy: {best_accuracy:.4f}")
    print(f"  最佳 Epoch: {best_epoch}")
    return model, best_accuracy


# ==================== 主函数 ====================
def main():
    safe_threads = configure_cpu_threads()
    get_memory_info()

    csv_path = os.path.join(base_dir, "dataset", "training_validation.csv")
    fasta_path = os.path.join(base_dir, "dataset", "training_validation_seqs")
    ind_csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    ind_fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    parser = argparse.ArgumentParser(description='两阶段训练：k-mer预训练 + 三路融合对比学习（含早停）')
    parser.add_argument('--input_fasta', default=fasta_path)
    parser.add_argument('--label_csv', default=csv_path)
    parser.add_argument('--ind_csv', default=ind_csv_path)
    parser.add_argument('--ind_fasta', default=ind_fasta_path)
    parser.add_argument('--output_path', default="results_two_stage(cnn357)")
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=32)

    # 训练参数
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='k-mer预训练最大轮数')
    parser.add_argument('--stage2_epochs', type=int, default=200, help='第二阶段最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--num_classes', type=int, default=9)

    # 联合损失参数
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='BCE损失权重 (1-alpha = 对比损失权重)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='对比学习温度参数')
    parser.add_argument('--contrastive_margin', type=float, default=1.0,
                        help='对比学习margin参数')
    parser.add_argument('--overlap_threshold', type=float, default=0.6,
                        help='标签重叠度阈值，大于此值视为正样本对')

    # ========== 新增：加载预训练k-mer权重 ==========
    parser.add_argument('--load_pretrained', action='store_true', default=False,
                        help='跳过预训练，直接加载已保存的k-mer编码器权重')
    parser.add_argument('--kmer_pretrain_path', type=str, default=None,
                        help='k-mer预训练权重路径（默认使用output_path下的权重）')

    parser.add_argument('--force_recompute', action='store_true')
    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)
    set_seed(42)
    device = torch.device(opt.device)

    print(f"\n{'=' * 60}")
    print("两阶段训练：k-mer预训练 + 三路融合对比学习（含早停）")
    print(f"模型架构: One-hot(CNN) + K-mer(3-6)(预训练MLP) + 7-mer(预训练MLP) → 拼接 → MLP分类头")
    print(f"  - CNN kernel_sizes: [3, 5, 7] → 192维")
    print(f"  - 融合维度: 192 + 256 + 256 = 704")
    print(f"  - 阶段1: 预训练k-mer编码器（仅BCE），patience={opt.patience}")
    print(f"  - 阶段2: 三路融合 + 对比学习 + BCE，patience={opt.patience}")
    print(f"  - K-mer编码器: 冻结 ❄️")
    print(f"预训练最大轮数: {opt.pretrain_epochs}")
    print(f"第二阶段最大轮数: {opt.stage2_epochs}")
    print(f"损失函数: {opt.alpha} * BCE + {1 - opt.alpha} * Contrastive")
    print(f"重叠度阈值: {opt.overlap_threshold}")
    print(f"温度参数: {opt.temperature}")
    print(f"{'=' * 60}\n")

    # 读取标签
    print("读取标签数据...")
    train_df = pd.read_csv(opt.label_csv)
    label_columns = train_df.columns[1:].tolist()
    print(f"标签列: {label_columns}")
    print(f"标签数量: {len(label_columns)}")

    train_labels_dict = {row[0]: row[1:1 + opt.num_classes].values.astype(np.float32)
                         for _, row in train_df.iterrows()}

    # 读取序列
    print("读取训练/验证序列...")
    fasta_records = read_nucleotide_sequences(opt.input_fasta)

    # 生成 one-hot 特征
    print("\n正在生成训练集one-hot特征...")
    train_onehot_features = {}
    for header, seq in tqdm(fasta_records, desc="生成one-hot特征"):
        train_onehot_features[header] = extract_onehot_features(seq, target_len=4000)
    train_ids = list(train_onehot_features.keys())
    print(f"成功生成 {len(train_onehot_features)} 条one-hot特征")

    # 保存缓存
    cache_dir = os.path.join(opt.output_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, 'train_onehot_4000.pkl'), 'wb') as f:
        pickle.dump(train_onehot_features, f)
    with open(os.path.join(cache_dir, 'train_ids.pkl'), 'wb') as f:
        pickle.dump(train_ids, f)

    # ========== 生成k-mer特征（分离3-6mer和7mer） ==========
    print("\n生成训练集3-6mer特征...")
    train_kmer_3_6_features = {}
    for name, seq in tqdm(fasta_records, desc="计算3-6mer"):
        combined = []
        for k in [3, 4, 5, 6]:
            headers = [''.join(combo) for combo in itertools.product('ACGT', repeat=k)]
            count = Counter()
            kmers = kmerArray(seq, k)
            count.update(kmers)
            if len(kmers) > 0:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [count.get(mer, 0) for mer in headers]
            combined.extend(code)
        train_kmer_3_6_features[name] = combined
    print(f"3-6mer特征维度: {len(combined)}")

    print("\n生成训练集7-mer特征...")
    train_kmer_7_features = {}
    headers_7 = [''.join(combo) for combo in itertools.product('ACGT', repeat=7)]
    for name, seq in tqdm(fasta_records, desc="计算7-mer"):
        count = Counter()
        kmers = kmerArray(seq, 7)
        count.update(kmers)
        if len(kmers) > 0:
            for key in count:
                count[key] = count[key] / len(kmers)
        code = [count.get(mer, 0) for mer in headers_7]
        train_kmer_7_features[name] = code
    print(f"7-mer特征维度: {len(headers_7)}")

    # 准备数据
    all_seq_ids = [sid for sid in train_ids if sid in train_labels_dict]
    valid_seq_ids = []
    for sid in all_seq_ids:
        if sid in train_kmer_3_6_features and sid in train_kmer_7_features:
            valid_seq_ids.append(sid)

    print(f"有效序列数: {len(valid_seq_ids)}")

    if len(valid_seq_ids) == 0:
        print("错误: 没有有效的序列!")
        sys.exit(1)

    train_ids_split, val_ids = train_test_split(valid_seq_ids, test_size=opt.val_ratio, random_state=42)
    print(f"训练集: {len(train_ids_split)} 条, 验证集: {len(val_ids)} 条")

    train_dataset = MultiLabelDataset(train_ids_split, train_onehot_features,
                                      train_kmer_3_6_features, train_kmer_7_features,
                                      train_labels_dict, opt.num_classes)
    val_dataset = MultiLabelDataset(val_ids, train_onehot_features,
                                    train_kmer_3_6_features, train_kmer_7_features,
                                    train_labels_dict, opt.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # ========== 判断是否跳过预训练 ==========
    if not opt.load_pretrained:
        # ========== 阶段1：预训练k-mer编码器 ==========
        print("\n" + "=" * 60)
        print("阶段1: 预训练K-mer编码器（仅BCE损失，含早停）")
        print("=" * 60)

        class KmerPretrainDataset(Dataset):
            def __init__(self, seq_ids, kmer_3_6_features, kmer_7_features, labels_dict, num_classes=9, kmer_type='kmer'):
                self.seq_ids = seq_ids
                self.kmer_3_6 = [kmer_3_6_features[sid] for sid in seq_ids]
                self.kmer_7 = [kmer_7_features[sid] for sid in seq_ids]
                self.labels_dict = labels_dict
                self.num_classes = num_classes
                self.kmer_type = kmer_type

            def __len__(self):
                return len(self.seq_ids)

            def __getitem__(self, idx):
                seq_id = self.seq_ids[idx]
                if self.kmer_type == 'kmer':
                    x = torch.tensor(self.kmer_3_6[idx], dtype=torch.float32)
                else:
                    x = torch.tensor(self.kmer_7[idx], dtype=torch.float32)
                labels = torch.tensor(self.labels_dict[seq_id][:self.num_classes], dtype=torch.float32)
                dummy_onehot = torch.zeros(1, 4)
                return dummy_onehot, x, x, labels, seq_id

        # 预训练3-6mer编码器
        print("\n>>> 预训练 3-6mer 编码器...")
        kmer_pretrain_dataset = KmerPretrainDataset(train_ids_split, train_kmer_3_6_features,
                                                    train_kmer_7_features, train_labels_dict,
                                                    opt.num_classes, kmer_type='kmer')
        kmer_pretrain_loader = DataLoader(kmer_pretrain_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        kmer_val_dataset = KmerPretrainDataset(val_ids, train_kmer_3_6_features,
                                               train_kmer_7_features, train_labels_dict,
                                               opt.num_classes, kmer_type='kmer')
        kmer_val_loader = DataLoader(kmer_val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

        kmer_encoder = KmerEncoderPretrain(num_classes=opt.num_classes).to(device)
        pretrain_kmer_encoder(kmer_encoder, kmer_pretrain_loader, kmer_val_loader, opt, device, encoder_type='kmer')

        # 预训练7-mer编码器
        print("\n>>> 预训练 7-mer 编码器...")
        sevenmer_pretrain_dataset = KmerPretrainDataset(train_ids_split, train_kmer_3_6_features,
                                                        train_kmer_7_features, train_labels_dict,
                                                        opt.num_classes, kmer_type='sevenmer')
        sevenmer_pretrain_loader = DataLoader(sevenmer_pretrain_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=0)
        sevenmer_val_dataset = KmerPretrainDataset(val_ids, train_kmer_3_6_features,
                                                   train_kmer_7_features, train_labels_dict,
                                                   opt.num_classes, kmer_type='sevenmer')
        sevenmer_val_loader = DataLoader(sevenmer_val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

        sevenmer_encoder = SevenMerEncoderPretrain(num_classes=opt.num_classes).to(device)
        pretrain_kmer_encoder(sevenmer_encoder, sevenmer_pretrain_loader, sevenmer_val_loader, opt, device,
                              encoder_type='sevenmer')

    # ========== 阶段2：三路融合 + 对比学习 ==========
    print("\n" + "=" * 60)
    print("阶段2: 三路融合 + 对比学习 + BCE 联合训练（含早停）")
    print("=" * 60)

    # 创建主模型
    model = MultiLabelModel(
        num_classes=opt.num_classes,
        dropout=0.15
    ).to(device)

    # ========== 加载预训练权重 ==========
    kmer_path = os.path.join(base_dir, "results_two_stage")

    kmer_state_path = os.path.join(kmer_path, 'kmer_pretrain_best.pt')
    if os.path.exists(kmer_state_path):
        kmer_state_dict = torch.load(kmer_state_path, map_location=device)
        model.kmer_encoder.load_state_dict(kmer_state_dict)
        print(f"✅ 3-6mer预训练权重已加载: {kmer_state_path}")
    else:
        print(f"⚠️ 警告: 3-6mer预训练权重未找到: {kmer_state_path}")

    sevenmer_state_path = os.path.join(kmer_path, 'sevenmer_pretrain_best.pt')
    if os.path.exists(sevenmer_state_path):
        sevenmer_state_dict = torch.load(sevenmer_state_path, map_location=device)
        model.sevenmer_encoder.load_state_dict(sevenmer_state_dict)
        print(f"✅ 7-mer预训练权重已加载: {sevenmer_state_path}")
    else:
        print(f"⚠️ 警告: 7-mer预训练权重未找到: {sevenmer_state_path}")

    # 冻结K-mer编码器
    model.freeze_kmer_encoders()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,} (K-mer编码器已冻结 ❄️)")

    # 第二阶段训练
    model, best_acc = train_stage2(model, train_loader, val_loader, opt, device)

    # ========== 独立测试集评估 ==========
    if os.path.exists(ind_csv_path) and os.path.exists(ind_fasta_path):
        print("\n" + "=" * 60)
        print("独立测试集评估")
        print("=" * 60)

        ind_label_df = pd.read_csv(ind_csv_path)
        ind_labels_dict = {row[0]: row[1:1 + opt.num_classes].values.astype(np.float32)
                           for _, row in ind_label_df.iterrows()}

        ind_fasta = read_nucleotide_sequences(ind_fasta_path)

        print("生成测试集one-hot特征...")
        ind_onehot_features = {}
        for header, seq in tqdm(ind_fasta, desc="生成测试集one-hot"):
            ind_onehot_features[header] = extract_onehot_features(seq, target_len=4000)

        print("生成测试集3-6mer特征...")
        ind_kmer_3_6_features = {}
        for name, seq in tqdm(ind_fasta, desc="计算3-6mer"):
            combined = []
            for k in [3, 4, 5, 6]:
                headers = [''.join(combo) for combo in itertools.product('ACGT', repeat=k)]
                count = Counter()
                kmers = kmerArray(seq, k)
                count.update(kmers)
                if len(kmers) > 0:
                    for key in count:
                        count[key] = count[key] / len(kmers)
                code = [count.get(mer, 0) for mer in headers]
                combined.extend(code)
            ind_kmer_3_6_features[name] = combined

        print("生成测试集7-mer特征...")
        ind_kmer_7_features = {}
        headers_7 = [''.join(combo) for combo in itertools.product('ACGT', repeat=7)]
        for name, seq in tqdm(ind_fasta, desc="计算7-mer"):
            count = Counter()
            kmers = kmerArray(seq, 7)
            count.update(kmers)
            if len(kmers) > 0:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [count.get(mer, 0) for mer in headers_7]
            ind_kmer_7_features[name] = code

        ind_valid_ids = [sid for sid, _ in ind_fasta
                         if sid in ind_labels_dict and sid in ind_onehot_features
                         and sid in ind_kmer_3_6_features and sid in ind_kmer_7_features]

        ind_dataset = MultiLabelDataset(ind_valid_ids, ind_onehot_features,
                                        ind_kmer_3_6_features, ind_kmer_7_features,
                                        ind_labels_dict, opt.num_classes)
        ind_loader = DataLoader(ind_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

        model_path = os.path.join(opt.output_path, 'best_stage2_model.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print("已加载第二阶段最佳模型")

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for onehot, kmer_3_6, kmer_7, labels, seq_ids in tqdm(ind_loader, desc="测试集预测"):
                onehot = onehot.to(device)
                kmer_3_6 = kmer_3_6.to(device)
                kmer_7 = kmer_7.to(device)
                labels = labels.to(device)

                logits = model(onehot, kmer_3_6, kmer_7)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_multilabel_metrics(all_labels, all_logits)

        print("\n独立测试集结果:")
        print("-" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(os.path.join(opt.output_path, 'final_results.txt'), 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("独立测试集结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"模型架构: One-hot(CNN) + K-mer(3-6)(预训练MLP) + 7-mer(预训练MLP) → 拼接 → MLP分类头\n")
            f.write(f"  - CNN kernel_sizes: [3, 5, 7] → 192维\n")
            f.write(f"  - 融合维度: 192 + 256 + 256 = 704\n")
            f.write(f"  - 阶段1: 预训练k-mer编码器（仅BCE），patience={opt.patience}\n")
            f.write(f"  - 阶段2: 三路融合 + 对比学习 + BCE，patience={opt.patience}\n")
            f.write(f"  - K-mer编码器: 冻结 ❄️\n")
            f.write(f"  - 分类头: 融合特征 → 256 → 64 → {opt.num_classes}\n")
            f.write(f"训练配置: pretrain_epochs={opt.pretrain_epochs}, stage2_epochs={opt.stage2_epochs}\n")
            f.write(f"联合损失: α={opt.alpha}, 1-α={1 - opt.alpha}\n")
            f.write(f"重叠度阈值: {opt.overlap_threshold}\n")
            f.write(f"Temperature: {opt.temperature}\n")
            f.write(f"Margin: {opt.contrastive_margin}\n")
            f.write("-" * 50 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"\n结果已保存到: {os.path.join(opt.output_path, 'final_results.txt')}")

    print("\n" + "=" * 60)
    print("全部训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()