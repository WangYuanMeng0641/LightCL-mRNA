import re, os, sys
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore')

# ========== 导入模型类 ==========
try:
    from cnn357 import (
        MultiLabelModel,
        MultiLabelDataset,
        extract_onehot_features,
        read_nucleotide_sequences,
        kmerArray,
        compute_multilabel_metrics
    )

    print("✅ 从 cnn357.py 导入成功")
except ImportError:
    try:
        from train import (
            MultiLabelModel,
            MultiLabelDataset,
            extract_onehot_features,
            read_nucleotide_sequences,
            kmerArray,
            compute_multilabel_metrics
        )

        print("✅ 从 train.py 导入成功")
    except ImportError:
        try:
            from main import (
                MultiLabelModel,
                MultiLabelDataset,
                extract_onehot_features,
                read_nucleotide_sequences,
                kmerArray,
                compute_multilabel_metrics
            )

            print("✅ 从 main.py 导入成功")
        except ImportError:
            print("❌ 无法导入模型类，请确保训练脚本在正确位置")
            sys.exit(1)


# ==================== 特征名称生成 ====================
def generate_full_feature_names():
    """
    生成704维融合特征的完整名称
    特征组: OnehotFeat (0-191), 3-6mer (192-447), 7-mer (448-703)
    """
    feature_names = []

    # ========== 1. One-hot序列特征 (0-191) ==========
    for i in range(192):
        feature_names.append(f"OnehotFeat_{i:03d}")

    # ========== 2. 3-6mer特征 (192-447) ==========
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_3_6_list = []
    for k in [3, 4, 5, 6]:
        for kmer in itertools.product(nucleotides, repeat=k):
            kmer_3_6_list.append(''.join(kmer))
    for i in range(min(256, len(kmer_3_6_list))):
        feature_names.append(f"Kmer_{kmer_3_6_list[i]}")

    # ========== 3. 7-mer特征 (448-703) ==========
    kmer_7_list = []
    for kmer in itertools.product(nucleotides, repeat=7):
        kmer_7_list.append(''.join(kmer))
    for i in range(min(256, len(kmer_7_list))):
        feature_names.append(f"Kmer_{kmer_7_list[i]}")

    return feature_names


def get_feature_group(idx):
    """判断704维特征属于哪个组"""
    if idx < 192:
        return "OnehotFeat"
    elif idx < 448:  # 192 + 256 = 448
        return "3-6mer"
    else:  # 448 + 256 = 704
        return "7-mer"


def get_feature_color(group_name):
    """根据特征组返回颜色"""
    colors = {
        "OnehotFeat": "#BBDEFB",  # 浅蓝
        "3-6mer": "#FFE0B2",  # 浅橙
        "7-mer": "#C8E6C9"  # 浅绿
    }
    return colors.get(group_name, "#E0E0E0")


# ==================== SHAP分析器 ====================
class SHAPAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.full_feature_names = generate_full_feature_names()

        print(f"生成 {len(self.full_feature_names)} 个融合特征名称 (704维)")
        print(f"特征组: OnehotFeat (0-191), 3-6mer (192-447), 7-mer (448-703)")
        print(f"  - OnehotFeat: 从One-hot序列通过CNN提取的192维特征")
        print(f"  - 3-6mer: 256维K-mer频次特征")
        print(f"  - 7-mer: 256维K-mer频次特征")
        print(f"使用解释器: PermutationExplainer")

    def prepare_fused_data(self, dataloader, max_samples=3000):
        """提取融合特征数据 (704维)"""
        print("准备融合特征数据...")
        all_fused = []
        all_labels = []
        all_seq_ids = []

        with torch.no_grad():
            for onehot, kmer_3_6, kmer_7, labels, seq_ids in tqdm(dataloader, desc="提取融合特征"):
                onehot = onehot.to(self.device)
                kmer_3_6 = kmer_3_6.to(self.device)
                kmer_7 = kmer_7.to(self.device)

                _, fused = self.model(onehot, kmer_3_6, kmer_7, return_features=True)

                all_fused.append(fused.cpu().numpy())
                all_labels.append(labels.numpy())
                all_seq_ids.extend(seq_ids)

        X_fused = np.vstack(all_fused)
        y = np.vstack(all_labels)

        print(f"总数据量: {X_fused.shape[0]} 条序列")
        print(f"融合特征形状: {X_fused.shape} (704维)")
        print(f"  - OnehotFeat (0-191): 192维")
        print(f"  - 3-6mer (192-447): 256维")
        print(f"  - 7-mer (448-703): 256维")

        if X_fused.shape[0] > max_samples:
            indices = np.random.choice(X_fused.shape[0], max_samples, replace=False)
            X_fused = X_fused[indices]
            y = y[indices]
            print(f"采样后数据量: {X_fused.shape[0]} 条序列（{max_samples}个样本）")

        return X_fused, y

    def get_prediction_from_fused(self, fused_data):
        """从融合特征获取预测概率"""
        fused_tensor = torch.tensor(fused_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model.classifier(fused_tensor)
        return torch.sigmoid(logits).cpu().numpy()

    def analyze_combined_features(self, X_fused, y, output_path, label_names=None, label_indices=None,
                                  top_k=15, bg_size=300, shap_samples=3000):
        """分析融合特征 - 三个组综合取前15个特征"""
        print("\n" + "=" * 70)
        print("分析融合特征SHAP值 (三个特征组合并)")
        print(f"特征组: OnehotFeat (0-191), 3-6mer (192-447), 7-mer (448-703)")
        print(f"从所有特征中综合取前 {top_k} 个最重要特征")
        print("=" * 70)

        if label_names is None:
            label_names = [f"Label_{i + 1}" for i in range(y.shape[1])]

        if label_indices is None:
            label_indices = list(range(y.shape[1]))

        # ========== 设置SHAP分析样本数 ==========
        total_samples = X_fused.shape[0]
        if shap_samples is None or shap_samples > total_samples:
            shap_samples = total_samples
        sample_size = min(shap_samples, total_samples)
        indices = np.random.choice(total_samples, sample_size, replace=False)
        X_sample = X_fused[indices]
        print(f"使用 {X_sample.shape[0]} 个样本进行SHAP分析")

        # ========== 设置背景数据 ==========
        bg_size = min(bg_size, X_sample.shape[0])
        bg_indices = np.random.choice(X_sample.shape[0], bg_size, replace=False)
        background = X_sample[bg_indices]
        print(f"背景数据: {bg_size} 个样本")

        num_features = X_sample.shape[1]
        min_evals = 2 * num_features + 1
        max_evals = max(2000, min_evals)
        print(f"特征数: {num_features}, 最小评估次数: {min_evals}, 实际使用: {max_evals}")

        for label_idx in label_indices:
            label_name = label_names[label_idx] if label_idx < len(label_names) else f"Label_{label_idx + 1}"
            print(f"\n{'=' * 50}")
            print(f"分析 {label_name}")
            print(f"{'=' * 50}")

            def predict_fn(data):
                return self.get_prediction_from_fused(data)[:, label_idx]

            try:
                explainer = shap.PermutationExplainer(predict_fn, background, max_evals=max_evals)
                shap_sample_size = min(300, X_sample.shape[0])
                shap_indices = np.random.choice(X_sample.shape[0], shap_sample_size, replace=False)
                X_shap = X_sample[shap_indices]

                shap_values = explainer(X_shap)
                shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values

                # ========== 综合所有特征取前15个 ==========
                self._save_combined_results(
                    shap_values_array, X_shap, label_name, output_path,
                    label_idx, top_k
                )

            except Exception as e:
                print(f"❌ SHAP分析失败: {e}")
                continue

    def _save_combined_results(self, shap_values_array, X_sample, label_name, output_path,
                               label_idx, top_k=15):
        """保存综合结果 - 所有特征取前15个，输出PNG和PDF，800 DPI"""

        # 计算所有特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

        # 取前top_k个特征
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_features = [self.full_feature_names[i] for i in sorted_indices]
        top_shap = mean_abs_shap[sorted_indices]
        top_groups = [get_feature_group(i) for i in sorted_indices]
        top_indices = sorted_indices

        # 提取对应的SHAP值和特征值
        combined_shap = shap_values_array[:, sorted_indices]
        combined_X = X_sample[:, sorted_indices]

        # 保存CSV
        csv_filename = f"shap_combined_{label_name}_top{top_k}.csv"
        df = pd.DataFrame({
            'Feature_Index': top_indices,
            'Feature_Name': top_features,
            'Feature_Group': top_groups,
            'Mean_Abs_SHAP': top_shap,
            'Rank': range(1, len(top_features) + 1)
        })
        df.to_csv(os.path.join(output_path, csv_filename), index=False)
        print(f"  已保存CSV: {csv_filename}")

        # 统计各组特征数量
        group_counts = {}
        for g in top_groups:
            group_counts[g] = group_counts.get(g, 0) + 1
        print(f"  特征组分布: {group_counts}")

        # ========== 绘制合并的SHAP图 ==========
        plt.figure(figsize=(14, 10))

        # 使用完整特征名称
        short_names = top_features.copy()

        # 使用shap绘制
        shap.summary_plot(
            combined_shap,
            combined_X,
            feature_names=short_names,
            plot_type="dot",
            show=False,
            max_display=top_k
        )

        # 添加标题
        plt.title(f"{label_name} - Top {top_k} Features",
                  fontsize=14, fontweight='bold')

        # ========== 删除图例 ==========
        # 不再添加图例

        plt.tight_layout()

        # 保存PNG (800 DPI)
        png_filename = f"shap_combined_{label_name}_top{top_k}.png"
        plt.savefig(os.path.join(output_path, png_filename), dpi=800, bbox_inches='tight')
        print(f"  已保存PNG (800 DPI): {png_filename}")

        # 保存PDF
        pdf_filename = f"shap_combined_{label_name}_top{top_k}.pdf"
        plt.savefig(os.path.join(output_path, pdf_filename), format='pdf', bbox_inches='tight')
        print(f"  已保存PDF: {pdf_filename}")

        plt.close()

        # 打印Top 10特征
        print(f"  Top {min(10, top_k)} 最重要特征:")
        for i in range(min(10, len(top_features))):
            group = top_groups[i]
            print(f"    {i + 1}. [{group}] {top_features[i]} (SHAP={top_shap[i]:.6f})")

# ==================== 主函数 ====================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ind_csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    ind_fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    parser = argparse.ArgumentParser(description='SHAP分析 - 加载训练好的模型 (704维)')
    parser.add_argument('--model_path', default='results_two_stage(cnn357)/best_stage2_model.pt',
                        help='训练好的模型路径')
    parser.add_argument('--output_path', default='shap_combined_results',
                        help='输出路径')
    parser.add_argument('--ind_csv', default=ind_csv_path,
                        help='独立测试集标签CSV')
    parser.add_argument('--ind_fasta', default=ind_fasta_path,
                        help='独立测试集FASTA')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--max_samples_fused', type=int, default=3000,
                        help='融合特征采样的最大样本数（用于SHAP分析）')
    parser.add_argument('--bg_size', type=int, default=300,
                        help='SHAP背景数据大小')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_k', type=int, default=15,
                        help='综合取前K个特征')
    opt = parser.parse_args()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    os.makedirs(opt.output_path, exist_ok=True)

    print("=" * 70)
    print("SHAP分析 - 加载训练好的模型 (704维)")
    print(f"模型路径: {opt.model_path}")
    print(f"输出路径: {opt.output_path}")
    print(f"背景数据大小: {opt.bg_size}")
    print(f"SHAP分析样本数: {opt.max_samples_fused}")
    print(f"综合取前 {opt.top_k} 个特征")
    print("=" * 70)

    # ========== 加载模型 ==========
    print(f"\n加载模型...")
    if not os.path.exists(opt.model_path):
        print(f"❌ 错误: 模型文件不存在 {opt.model_path}")
        sys.exit(1)

    model = MultiLabelModel(num_classes=opt.num_classes).to(opt.device)
    model.load_state_dict(torch.load(opt.model_path, map_location=opt.device))
    model.eval()
    print("✅ 模型加载成功")

    # ========== 准备数据 ==========
    print(f"\n准备独立测试集数据...")

    ind_label_df = pd.read_csv(opt.ind_csv)
    label_columns = ind_label_df.columns[1:1 + opt.num_classes].tolist()
    print(f"标签列: {label_columns}")

    ind_labels_dict = {row[0]: row[1:1 + opt.num_classes].values.astype(np.float32)
                       for _, row in ind_label_df.iterrows()}

    ind_fasta = read_nucleotide_sequences(opt.ind_fasta)
    print(f"读取到 {len(ind_fasta)} 条序列")

    # ========== 提取特征 ==========
    print("\n提取特征...")

    print("  生成One-hot特征...")
    ind_onehot_features = {}
    for header, seq in tqdm(ind_fasta, desc="生成One-hot"):
        ind_onehot_features[header] = extract_onehot_features(seq, target_len=4000)

    print("  生成3-6mer特征...")
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

    print("  生成7-mer特征...")
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

    # ========== 对齐数据 ==========
    print("\n对齐数据...")
    ind_valid_ids = [sid for sid, _ in ind_fasta
                     if sid in ind_labels_dict and sid in ind_onehot_features
                     and sid in ind_kmer_3_6_features and sid in ind_kmer_7_features]

    print(f"有效序列数: {len(ind_valid_ids)}")

    if len(ind_valid_ids) == 0:
        print("❌ 错误: 没有有效的序列!")
        sys.exit(1)

    ind_dataset = MultiLabelDataset(ind_valid_ids, ind_onehot_features,
                                    ind_kmer_3_6_features, ind_kmer_7_features,
                                    ind_labels_dict, opt.num_classes)
    ind_loader = DataLoader(ind_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # ========== 计算独立测试集指标 ==========
    print("\n计算独立测试集指标...")
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for onehot, kmer_3_6, kmer_7, labels, seq_ids in tqdm(ind_loader, desc="预测"):
            onehot = onehot.to(opt.device)
            kmer_3_6 = kmer_3_6.to(opt.device)
            kmer_7 = kmer_7.to(opt.device)
            labels = labels.to(opt.device)

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

    # ========== 创建SHAP分析器 ==========
    analyzer = SHAPAnalyzer(model, opt.device)

    # ========== 分析融合特征 (三个组合并) ==========
    print("\n" + "=" * 70)
    print("分析融合特征 (OnehotFeat + 3-6mer + 7-mer 综合Top15)")
    print("=" * 70)
    X_fused, y = analyzer.prepare_fused_data(ind_loader, max_samples=opt.max_samples_fused)
    analyzer.analyze_combined_features(X_fused, y, opt.output_path, label_columns,
                                       top_k=opt.top_k, bg_size=opt.bg_size,
                                       shap_samples=opt.max_samples_fused)

    print("\n" + "=" * 70)
    print("SHAP分析完成！")
    print(f"结果保存在: {opt.output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()