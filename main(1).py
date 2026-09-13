import os
import sys
import re
import itertools
import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import jaccard_score, hamming_loss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# ========== 从 new.py 导入模型和相关函数 ==========
from new import (
    MultiLabelModel,
    MultiLabelDataset,
    extract_onehot_features,
    read_nucleotide_sequences,
    kmerArray,
    compute_multilabel_metrics
)


def load_model(model_path, num_classes=6, device='cuda'):
    """加载训练好的模型"""
    model = MultiLabelModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def prepare_data_with_folds(fasta_path, csv_path, n_folds=5, batch_size=32):
    """
    准备数据并生成5折交叉验证的索引
    返回: 5折的 (train_ids, val_ids) 列表
    """
    print("\n准备数据...")

    # 读取标签 - 从FASTA文件中提取标签
    print("从FASTA文件中读取标签数据...")
    fasta_records = read_nucleotide_sequences(fasta_path)

    # 标签列（6个定位）
    label_columns = ['nucleus', 'exosome', 'cytosol', 'ribosome', 'membrane', 'endoplasmic reticulum']

    # 从header中解析标签
    labels_dict = {}
    for header, seq in fasta_records:
        if ',' in header:
            label_str = header.split(',')[0]
        else:
            label_str = header

        # 将标签字符串转换为6维向量
        labels = np.zeros(len(label_columns), dtype=np.float32)
        for i, char in enumerate(label_str):
            if i < len(label_columns) and char == '1':
                labels[i] = 1.0

        labels_dict[header] = labels

    print(f"标签字典大小: {len(labels_dict)}")
    print(f"读取了 {len(fasta_records)} 条序列")

    # 生成one-hot特征
    print("生成one-hot特征...")
    onehot_features = {}
    for header, seq in tqdm(fasta_records, desc="One-hot"):
        onehot_features[header] = extract_onehot_features(seq, target_len=4000)

    # 生成3-6mer特征
    print("生成3-6mer特征...")
    kmer_3_6_features = {}
    for name, seq in tqdm(fasta_records, desc="3-6mer"):
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
        kmer_3_6_features[name] = combined

    # 生成7-mer特征
    print("生成7-mer特征...")
    kmer_7_features = {}
    headers_7 = [''.join(combo) for combo in itertools.product('ACGT', repeat=7)]
    for name, seq in tqdm(fasta_records, desc="7-mer"):
        count = Counter()
        kmers = kmerArray(seq, 7)
        count.update(kmers)
        if len(kmers) > 0:
            for key in count:
                count[key] = count[key] / len(kmers)
        code = [count.get(mer, 0) for mer in headers_7]
        kmer_7_features[name] = code

    # 验证有效序列
    valid_ids = []
    for sid, _ in fasta_records:
        if sid in labels_dict and sid in onehot_features and sid in kmer_3_6_features and sid in kmer_7_features:
            valid_ids.append(sid)

    print(f"有效序列数: {len(valid_ids)}")

    if len(valid_ids) == 0:
        print("错误: 没有有效的序列!")
        sys.exit(1)

    # 创建完整数据集
    full_dataset = MultiLabelDataset(valid_ids, onehot_features, kmer_3_6_features, kmer_7_features, labels_dict, 6)

    # 生成5折交叉验证的索引
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_indices = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(valid_ids)):
        fold_indices.append({
            'fold': fold + 1,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'train_ids': [valid_ids[i] for i in train_idx],
            'val_ids': [valid_ids[i] for i in val_idx]
        })
        print(f"Fold {fold + 1}: 训练集 {len(train_idx)} 条, 验证集 {len(val_idx)} 条")

    return full_dataset, valid_ids, label_columns, fold_indices


def compute_per_class_metrics(y_true, y_pred_probs, label_names):
    """计算每个类别的AUC、APR、MCC"""
    n_classes = y_true.shape[1]

    results = {
        'class': [],
        'auc': [],
        'apr': [],
        'mcc': []
    }

    for i in range(n_classes):
        class_name = label_names[i] if i < len(label_names) else f"Class_{i + 1}"

        try:
            auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except:
            auc = float('nan')

        try:
            apr = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        except:
            apr = float('nan')

        try:
            y_pred_binary = (y_pred_probs[:, i] > 0.5).astype(int)
            mcc = matthews_corrcoef(y_true[:, i], y_pred_binary)
        except:
            mcc = float('nan')

        results['class'].append(class_name)
        results['auc'].append(auc)
        results['apr'].append(apr)
        results['mcc'].append(mcc)

    return results


def compute_multilabel_metrics_eval(y_true, y_pred_probs):
    """计算多标签整体评估指标（5个）"""
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Aiming
    aiming = np.mean([
        np.sum(y_true[i] * y_pred[i]) / np.sum(y_pred[i])
        if np.sum(y_pred[i]) > 0 else 0
        for i in range(len(y_true))
    ])

    # Coverage
    coverage = np.mean([
        np.sum(y_true[i] * y_pred[i]) / np.sum(y_true[i])
        if np.sum(y_true[i]) > 0 else 0
        for i in range(len(y_true))
    ])

    # Accuracy (Jaccard)
    accuracy = jaccard_score(y_true, y_pred, average='samples')

    # Absolute-True
    absolute_true = np.mean([np.array_equal(y_true[i], y_pred[i]) for i in range(len(y_true))])

    # Absolute-False (Hamming Loss)
    absolute_false = hamming_loss(y_true, y_pred)

    return {
        'aiming': aiming,
        'coverage': coverage,
        'accuracy': accuracy,
        'absolute_true': absolute_true,
        'absolute_false': absolute_false
    }


def evaluate_on_subset(model, dataset, indices, device, batch_size=32):
    """
    在指定的数据子集上评估模型
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_pred_probs = []

    model.eval()
    with torch.no_grad():
        for onehot, kmer_3_6, kmer_7, labels, seq_ids in loader:
            onehot = onehot.to(device)
            kmer_3_6 = kmer_3_6.to(device)
            kmer_7 = kmer_7.to(device)

            logits = model(onehot, kmer_3_6, kmer_7)
            pred_probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_pred_probs.append(pred_probs.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred_probs = np.vstack(all_pred_probs)

    return y_true, y_pred_probs


def main():
    # ========== 配置 ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 5折模型路径
    model_dir = os.path.join(base_dir, "results")
    model_prefix = "best_stage2_fold"
    n_folds = 5

    # 数据路径
    fasta_path = os.path.join(base_dir, "data", "modified_multilabel_seq_6labels.fasta")
    csv_path = os.path.join(base_dir, "data", "modified_multilabel_seq_6labels.fasta")

    # 标签名称（6个定位）
    label_names = ['nucleus', 'exosome', 'cytosol', 'ribosome', 'membrane', 'endoplasmic reticulum']
    num_classes = len(label_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型目录: {model_dir}")

    # ========== 检查模型文件是否存在 ==========
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f"{model_prefix}{fold}.pt")
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_path}")
            print("请先运行训练脚本训练模型")
            return
    print("✅ 所有5个fold模型文件存在")

    # ========== 准备数据和5折索引 ==========
    full_dataset, valid_ids, label_columns, fold_indices = prepare_data_with_folds(
        fasta_path, csv_path, n_folds=n_folds
    )

    # ========== 存储每个fold的指标 ==========
    fold_aiming_scores = []
    fold_coverage_scores = []
    fold_accuracy_multilabel_scores = []
    fold_absolute_true_scores = []
    fold_absolute_false_scores = []

    # 存储每个fold的每个类别AUC/APR/MCC
    all_fold_per_class_results = []

    # ========== 对每个fold进行评估 ==========
    print("\n" + "=" * 60)
    print("对每个fold的验证集进行评估")
    print("=" * 60)

    for fold_info in fold_indices:
        fold = fold_info['fold']
        val_idx = fold_info['val_idx']

        model_path = os.path.join(model_dir, f"{model_prefix}{fold}.pt")

        print(f"\n{'=' * 60}")
        print(f"Fold {fold}/{n_folds}")
        print(f"验证集样本数: {len(val_idx)}")
        print(f"模型: {model_path}")
        print(f"{'=' * 60}")

        # 加载模型
        model = load_model(model_path, num_classes, device)

        # 在验证集上评估
        y_true, y_pred_probs = evaluate_on_subset(model, full_dataset, val_idx, device)

        # ========== 计算多标签5个指标 ==========
        multilabel_metrics = compute_multilabel_metrics_eval(y_true, y_pred_probs)
        fold_aiming_scores.append(multilabel_metrics['aiming'])
        fold_coverage_scores.append(multilabel_metrics['coverage'])
        fold_accuracy_multilabel_scores.append(multilabel_metrics['accuracy'])
        fold_absolute_true_scores.append(multilabel_metrics['absolute_true'])
        fold_absolute_false_scores.append(multilabel_metrics['absolute_false'])

        # ========== 计算每个类别的AUC/APR/MCC ==========
        per_class_results = compute_per_class_metrics(y_true, y_pred_probs, label_names)
        all_fold_per_class_results.append({
            'fold': fold,
            'results': per_class_results
        })

        # ========== 打印结果 ==========
        print(f"\nFold {fold} 每个类别指标 (验证集):")
        print("{:<20s} {:>12s} {:>12s} {:>12s}".format("Class", "AUC", "APR", "MCC"))
        print("-" * 60)
        for i, class_name in enumerate(per_class_results['class']):
            print("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                class_name,
                per_class_results['auc'][i],
                per_class_results['apr'][i],
                per_class_results['mcc'][i]
            ))
        print("-" * 60)
        print("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            "Average",
            np.mean(per_class_results['auc']),
            np.mean(per_class_results['apr']),
            np.mean(per_class_results['mcc'])
        ))

        print(f"\nFold {fold} 多标签指标 (验证集):")
        print(f"  Aiming: {multilabel_metrics['aiming']:.4f}")
        print(f"  Coverage: {multilabel_metrics['coverage']:.4f}")
        print(f"  Accuracy: {multilabel_metrics['accuracy']:.4f}")
        print(f"  Absolute_True: {multilabel_metrics['absolute_true']:.4f}")
        print(f"  Absolute_False: {multilabel_metrics['absolute_false']:.4f}")

    # ========== 计算5折平均结果 ==========
    print("\n" + "=" * 60)
    print("5折交叉验证 - 平均结果")
    print("=" * 60)

    print("\n多标签5个指标 (5折平均):")
    print(f'  Aiming: {np.mean(fold_aiming_scores):.4f} ± {np.std(fold_aiming_scores):.4f}')
    print(f'  Coverage: {np.mean(fold_coverage_scores):.4f} ± {np.std(fold_coverage_scores):.4f}')
    print(f'  Accuracy: {np.mean(fold_accuracy_multilabel_scores):.4f} ± {np.std(fold_accuracy_multilabel_scores):.4f}')
    print(f'  Absolute_True: {np.mean(fold_absolute_true_scores):.4f} ± {np.std(fold_absolute_true_scores):.4f}')
    print(f'  Absolute_False: {np.mean(fold_absolute_false_scores):.4f} ± {np.std(fold_absolute_false_scores):.4f}')

    # ========== 计算每个类别AUC/APR/MCC的5折平均 ==========
    print("\n每个类别AUC/APR/MCC的5折平均 (验证集):")
    print("{:<20s} {:>12s} {:>12s} {:>12s}".format("Class", "AUC", "APR", "MCC"))
    print("-" * 60)

    n_classes = len(label_names)
    avg_auc_list = []
    avg_apr_list = []
    avg_mcc_list = []

    for i in range(n_classes):
        class_name = label_names[i]
        aucs = [fold['results']['auc'][i] for fold in all_fold_per_class_results]
        aprs = [fold['results']['apr'][i] for fold in all_fold_per_class_results]
        mccs = [fold['results']['mcc'][i] for fold in all_fold_per_class_results]

        avg_auc = np.mean(aucs)
        avg_apr = np.mean(aprs)
        avg_mcc = np.mean(mccs)

        avg_auc_list.append(avg_auc)
        avg_apr_list.append(avg_apr)
        avg_mcc_list.append(avg_mcc)

        print("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(class_name, avg_auc, avg_apr, avg_mcc))

    print("-" * 60)
    print("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        "Average", np.mean(avg_auc_list), np.mean(avg_apr_list), np.mean(avg_mcc_list)
    ))

    # ========== 保存结果 ==========
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    output_dir = os.path.join(model_dir, "evaluation_5fold_val")
    os.makedirs(output_dir, exist_ok=True)

    # 保存每个fold的每个类别指标
    all_rows = []
    for fold_data in all_fold_per_class_results:
        fold = fold_data['fold']
        results = fold_data['results']
        for i, class_name in enumerate(results['class']):
            all_rows.append({
                'fold': fold,
                'class': class_name,
                'auc': results['auc'][i],
                'apr': results['apr'][i],
                'mcc': results['mcc'][i]
            })
    fold_class_df = pd.DataFrame(all_rows)
    fold_class_df.to_csv(os.path.join(output_dir, 'per_fold_class_metrics.csv'), index=False)
    print(f"✅ 已保存: {os.path.join(output_dir, 'per_fold_class_metrics.csv')}")

    # 保存每个类别平均指标
    avg_class_df = pd.DataFrame({
        'class': label_names + ['Average'],
        'auc': avg_auc_list + [np.mean(avg_auc_list)],
        'apr': avg_apr_list + [np.mean(avg_apr_list)],
        'mcc': avg_mcc_list + [np.mean(avg_mcc_list)]
    })
    avg_class_df.to_csv(os.path.join(output_dir, 'avg_class_metrics_5fold.csv'), index=False)
    print(f"✅ 已保存: {os.path.join(output_dir, 'avg_class_metrics_5fold.csv')}")

    # 保存多标签指标
    multilabel_df = pd.DataFrame({
        'fold': list(range(1, n_folds + 1)),
        'aiming': fold_aiming_scores,
        'coverage': fold_coverage_scores,
        'accuracy': fold_accuracy_multilabel_scores,
        'absolute_true': fold_absolute_true_scores,
        'absolute_false': fold_absolute_false_scores
    })
    multilabel_df.loc['Mean'] = multilabel_df.mean()
    multilabel_df.loc['Std'] = multilabel_df.std()
    multilabel_df.to_csv(os.path.join(output_dir, 'multilabel_metrics_5fold.csv'), index=True)
    print(f"✅ 已保存: {os.path.join(output_dir, 'multilabel_metrics_5fold.csv')}")

    # 保存汇总文本
    with open(os.path.join(output_dir, 'summary_5fold_results.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("5折交叉验证结果汇总 (验证集)\n")
        f.write("=" * 60 + "\n\n")

        f.write("多标签5个指标 (5折平均):\n")
        f.write(f"  Aiming: {np.mean(fold_aiming_scores):.4f} ± {np.std(fold_aiming_scores):.4f}\n")
        f.write(f"  Coverage: {np.mean(fold_coverage_scores):.4f} ± {np.std(fold_coverage_scores):.4f}\n")
        f.write(f"  Accuracy: {np.mean(fold_accuracy_multilabel_scores):.4f} ± {np.std(fold_accuracy_multilabel_scores):.4f}\n")
        f.write(f"  Absolute_True: {np.mean(fold_absolute_true_scores):.4f} ± {np.std(fold_absolute_true_scores):.4f}\n")
        f.write(f"  Absolute_False: {np.mean(fold_absolute_false_scores):.4f} ± {np.std(fold_absolute_false_scores):.4f}\n\n")

        f.write("每个类别AUC/APR/MCC (5折平均):\n")
        f.write("{:<20s} {:>12s} {:>12s} {:>12s}\n".format("Class", "AUC", "APR", "MCC"))
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(label_names):
            f.write("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}\n".format(
                class_name, avg_auc_list[i], avg_apr_list[i], avg_mcc_list[i]
            ))
        f.write("-" * 60 + "\n")
        f.write("{:<20s} {:>12.4f} {:>12.4f} {:>12.4f}\n".format(
            "Average", np.mean(avg_auc_list), np.mean(avg_apr_list), np.mean(avg_mcc_list)
        ))

    print(f"✅ 已保存: {os.path.join(output_dir, 'summary_5fold_results.txt')}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()