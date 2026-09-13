import os
import sys
import re
import itertools
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import IntegratedGradients

# ========== 导入你的模型 ==========
from train import (
    MultiLabelModel,
    MultiLabelDataset,
    extract_onehot_features,
    read_nucleotide_sequences,
    kmerArray,
    compute_multilabel_metrics
)


class IGMotifAnalyzer:
    def __init__(self, model_path, output_dir="ig_motif_window_results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.label_names = ['Exosome', 'Nucleus', 'Nucleoplasm', 'Chromatin',
                           'Cytoplasm', 'Nucleolus', 'Cytosol', 'Membrane', 'Ribosome']
        self.nucleotides = ['A', 'C', 'G', 'T']

        self.model = self._load_model(model_path)
        self.sequence_cache = {}

    def _load_model(self, model_path):
        print(f"加载模型: {model_path}")
        model = MultiLabelModel(num_classes=9).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        print("✅ 模型加载成功")
        return model

    def prepare_data(self, fasta_path, csv_path, batch_size=8, max_samples=None):
        """准备数据"""
        print("\n准备数据...")

        label_df = pd.read_csv(csv_path)
        labels_dict = {row[0]: row[1:1 + 9].values.astype(np.float32)
                       for _, row in label_df.iterrows()}

        fasta_records = read_nucleotide_sequences(fasta_path)

        self.sequence_cache = {header: seq for header, seq in fasta_records}

        print("生成one-hot特征...")
        onehot_features = {}
        for header, seq in tqdm(fasta_records, desc="One-hot"):
            onehot_features[header] = extract_onehot_features(seq, target_len=4000)

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

        valid_ids = [sid for sid, _ in fasta_records
                     if sid in labels_dict and sid in onehot_features
                     and sid in kmer_3_6_features and sid in kmer_7_features]

        print(f"有效序列数: {len(valid_ids)}")

        if max_samples is not None and len(valid_ids) > max_samples:
            import random
            random.seed(42)
            valid_ids = random.sample(valid_ids, max_samples)
            print(f"采样后: {len(valid_ids)}")
        else:
            print(f"使用全部 {len(valid_ids)} 条序列")

        dataset = MultiLabelDataset(valid_ids, onehot_features,
                                    kmer_3_6_features, kmer_7_features,
                                    labels_dict, 9)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return loader, labels_dict, valid_ids

    def compute_ig_scores(self, dataloader, target_class_idx, n_steps=25):
        """计算IG分数（位置级别）"""
        label_name = self.label_names[target_class_idx]
        print(f"\n计算 {label_name} 的IG分数...")

        def predict_func(onehot, kmer_3_6, kmer_7):
            logits = self.model(onehot, kmer_3_6, kmer_7)
            return torch.sigmoid(logits)[:, target_class_idx]

        ig = IntegratedGradients(predict_func)

        all_attributions = []
        all_seq_ids = []
        all_onehot = []
        all_scores = []

        self.model.eval()

        for batch in tqdm(dataloader, desc="计算IG"):
            onehot, kmer_3_6, kmer_7, labels, seq_ids = batch
            onehot = onehot.to(self.device)
            kmer_3_6 = kmer_3_6.to(self.device)
            kmer_7 = kmer_7.to(self.device)

            with torch.no_grad():
                logits = self.model(onehot, kmer_3_6, kmer_7)
                pred_probs = torch.sigmoid(logits)[:, target_class_idx]

            mask = (labels[:, target_class_idx] == 1) & (pred_probs > 0.5)
            if mask.sum() == 0:
                continue

            onehot_pos = onehot[mask]
            kmer_3_6_pos = kmer_3_6[mask]
            kmer_7_pos = kmer_7[mask]
            seq_ids_pos = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
            pred_scores = pred_probs[mask].cpu().numpy()

            try:
                attributions = ig.attribute(
                    inputs=onehot_pos,
                    target=None,
                    n_steps=n_steps,
                    additional_forward_args=(kmer_3_6_pos, kmer_7_pos),
                    internal_batch_size=4
                )

                all_attributions.append(attributions.cpu().detach().numpy())
                all_seq_ids.extend(seq_ids_pos)
                all_onehot.append(onehot_pos.cpu().numpy())
                all_scores.extend(pred_scores)

            except Exception as e:
                print(f"  ⚠️ IG计算失败: {e}")
                continue

        if len(all_attributions) == 0:
            print(f"  ⚠️ {label_name} 没有正样本")
            return None, None, None, None

        all_attributions = np.vstack(all_attributions)
        all_onehot = np.vstack(all_onehot)

        ig_scores = np.abs(all_attributions).sum(axis=2)

        print(f"  ✅ 计算了 {len(all_attributions)} 个样本的IG（位置级别）")

        return ig_scores, all_seq_ids, all_onehot, np.array(all_scores)

    def compute_window_scores(self, ig_scores, window_size=6):
        """将位置IG分数转换为滑动窗口IG分数"""
        print(f"\n转换为滑动{window_size}-mer窗口分数...")

        window_scores_list = []

        for scores in ig_scores:
            window_scores = np.convolve(scores, np.ones(window_size), 'valid')
            window_scores_list.append(window_scores)

        window_scores = np.array(window_scores_list)

        print(f"  ✅ 窗口分数形状: {window_scores.shape}")

        return window_scores

    def extract_motifs_with_scores(self, window_scores, all_onehot, all_seq_ids, all_scores,
                                   window_size=6, top_k=5):
        """从滑动窗口分数提取motif"""
        print(f"\n提取motif (滑动{window_size}-mer)...")

        all_motif_info = []

        for scores, onehot, seq_id, pred_score in zip(window_scores, all_onehot, all_seq_ids, all_scores):
            if len(scores) < 1:
                continue

            top_positions = np.argsort(scores)[-top_k:][::-1]

            for pos in top_positions:
                fragment = onehot[pos:pos+window_size]

                motif_str = ''
                for base_vec in fragment:
                    if np.sum(base_vec) > 0:
                        idx = np.argmax(base_vec)
                        motif_str += self.nucleotides[idx]
                    else:
                        motif_str += 'N'

                all_motif_info.append({
                    'seq_id': seq_id,
                    'position': pos,
                    'motif': motif_str,
                    'window_score': float(scores[pos]),
                    'pred_score': float(pred_score)
                })

        print(f"  ✅ 提取了 {len(all_motif_info)} 个{window_size}-mer motif")

        return all_motif_info

    def encode_motif(self, motif):
        """将motif编码为向量用于聚类"""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoding = np.zeros(len(motif) * 4, dtype=np.float32)
        for i, base in enumerate(motif):
            if base in base_map:
                encoding[i * 4 + base_map[base]] = 1
        return encoding

    def cluster_motifs(self, motif_info, similarity_threshold=0.4, min_cluster_size=3):
        """对motif进行聚类"""
        if len(motif_info) == 0:
            return []

        print(f"\n聚类motif (阈值={similarity_threshold})...")

        encoded = np.array([self.encode_motif(m['motif']) for m in motif_info])
        similarity = cosine_similarity(encoded)
        distance = 1 - similarity

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric='precomputed',
            linkage='average'
        )

        try:
            labels = clustering.fit_predict(distance)
        except Exception as e:
            print(f"  ⚠️ 聚类失败: {e}")
            labels = np.arange(len(motif_info))

        clusters = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:
                continue

            indices = np.where(labels == label)[0]
            if len(indices) < min_cluster_size:
                continue

            cluster_motifs = [motif_info[i] for i in indices]
            motifs = [m['motif'] for m in cluster_motifs]
            scores = [m['window_score'] for m in cluster_motifs]

            best_idx = np.argmax(scores)

            clusters.append({
                'cluster_id': len(clusters),
                'size': len(cluster_motifs),
                'motifs': motifs,
                'scores': scores,
                'representative': motifs[best_idx],
                'representative_score': scores[best_idx],
                'avg_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'items': cluster_motifs
            })

        clusters.sort(key=lambda x: x['avg_score'], reverse=True)

        print(f"  ✅ 聚类完成: {len(clusters)} 个聚类")

        return clusters

    def compute_pfm_from_cluster(self, cluster, window_size=6):
        """从聚类计算PFM"""
        pfm = np.zeros((window_size, 4))

        for motif in cluster['motifs']:
            for pos, base in enumerate(motif):
                if base in self.nucleotides:
                    idx = self.nucleotides.index(base)
                    pfm[pos, idx] += 1

        col_sums = pfm.sum(axis=1, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        pfm = pfm / col_sums

        return pfm.T  # [4, window_size]

    def save_cluster_meme(self, clusters, label_name, window_size=6):
        """保存聚类结果为MEME格式"""
        if len(clusters) == 0:
            return

        class_dir = self.class_output_dirs.get(label_name, self.output_dir)
        filepath = os.path.join(class_dir, f'motif_clusters.meme')

        with open(filepath, 'w') as f:
            f.write("MEME version 4\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")

            for cluster in clusters:
                pfm = self.compute_pfm_from_cluster(cluster, window_size)

                motif_name = f"{label_name}_Cluster{cluster['cluster_id']+1}"
                f.write(f"MOTIF {motif_name}\n")
                f.write(f"letter-probability matrix: alength= 4 w= {window_size} nsites= {cluster['size']}\n")
                f.write(f"# Consensus: {cluster['representative']}\n")
                f.write(f"# Avg IG score: {cluster['avg_score']:.6f}\n")

                for col in range(window_size):
                    for row in range(4):
                        f.write(f"{pfm[row, col]:.6f} ")
                    f.write("\n")
                f.write("\n")

        print(f"  ✅ 已保存MEME: {filepath}")

    def save_cluster_csv(self, clusters, label_name):
        """保存聚类结果到CSV"""
        if len(clusters) == 0:
            return

        class_dir = self.class_output_dirs.get(label_name, self.output_dir)
        rows = []
        for cluster in clusters:
            for motif, score in zip(cluster['motifs'], cluster['scores']):
                rows.append({
                    'cluster_id': cluster['cluster_id'] + 1,
                    'cluster_size': cluster['size'],
                    'motif': motif,
                    'ig_score': score,
                    'representative': cluster['representative'],
                    'avg_cluster_score': cluster['avg_score']
                })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(class_dir, f'motif_clusters.csv'), index=False)
        print(f"  ✅ 已保存CSV: {class_dir}/motif_clusters.csv")

    def _save_unclustered_motifs(self, motif_info, label_name):
        """保存未聚类的motif"""
        class_dir = self.class_output_dirs.get(label_name, self.output_dir)
        df = pd.DataFrame(motif_info)
        df.to_csv(os.path.join(class_dir, f'unclustered_motifs.csv'), index=False)
        print(f"  ✅ 已保存未聚类motif: {class_dir}/unclustered_motifs.csv")

    def run_full_analysis(self, fasta_path, csv_path,
                         batch_size=4, max_samples=100, n_steps=25,
                         similarity_threshold=0.4, min_cluster_size=3,
                         top_k_per_seq=5, window_size=6):
        """运行完整的IG motif分析（滑动窗口），只输出CSV和MEME"""
        print("=" * 60)
        print("Integrated Gradients Motif分析（滑动窗口）")
        print(f"样本数: {max_samples}, IG步数: {n_steps}")
        print(f"窗口大小: {window_size}bp")
        print(f"聚类阈值: {similarity_threshold}, 最小聚类大小: {min_cluster_size}")
        print("=" * 60)

        dataloader, labels_dict, valid_ids = self.prepare_data(
            fasta_path, csv_path, batch_size, max_samples
        )

        self.class_output_dirs = {}
        for label in self.label_names:
            class_dir = os.path.join(self.output_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            self.class_output_dirs[label] = class_dir

        for class_idx in range(9):
            label_name = self.label_names[class_idx]
            print(f"\n{'='*50}")
            print(f"分析类别: {label_name} ({class_idx+1}/9)")
            print(f"{'='*50}")

            ig_scores, seq_ids, all_onehot, all_scores = self.compute_ig_scores(
                dataloader, class_idx, n_steps=n_steps
            )

            if ig_scores is None:
                print(f"  ⚠️ {label_name} 没有正样本，跳过")
                continue

            window_scores = self.compute_window_scores(ig_scores, window_size=window_size)

            motif_info = self.extract_motifs_with_scores(
                window_scores, all_onehot, seq_ids, all_scores,
                window_size=window_size, top_k=top_k_per_seq
            )

            if len(motif_info) == 0:
                print(f"  ⚠️ {label_name} 没有提取到motif")
                continue

            clusters = self.cluster_motifs(
                motif_info,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size
            )

            if len(clusters) == 0:
                print(f"  ⚠️ {label_name} 聚类结果为空")
                self._save_unclustered_motifs(motif_info, label_name)
                continue

            self.save_cluster_csv(clusters, label_name)
            self.save_cluster_meme(clusters, label_name, window_size=window_size)

            print(f"\n  聚类统计 ({label_name}):")
            print(f"    总聚类数: {len(clusters)}")
            print(f"    总motif数: {sum(c['size'] for c in clusters)}")
            for i, cluster in enumerate(clusters[:3]):
                print(f"    聚类 {i+1}: {cluster['size']}个motif, "
                      f"代表序列: {cluster['representative']}, "
                      f"平均分数: {cluster['avg_score']:.6f}")

        print("\n" + "=" * 60)
        print(f"✅ 完成！结果保存在: {self.output_dir}")
        print("=" * 60)


# ========== 主函数 ==========
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "results_two_stage", "best_stage2_model.pt")
    csv_path = os.path.join(base_dir, "dataset", "independent.csv")
    fasta_path = os.path.join(base_dir, "dataset", "independent_seqs")

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    analyzer = IGMotifAnalyzer(
        model_path=model_path,
        output_dir="ig_motif_window_results"
    )

    analyzer.run_full_analysis(
        fasta_path=fasta_path,
        csv_path=csv_path,
        batch_size=4,
        max_samples=None,
        n_steps=25,
        similarity_threshold=0.7,
        min_cluster_size=5,
        top_k_per_seq=3,
        window_size=6
    )


if __name__ == "__main__":
    main()