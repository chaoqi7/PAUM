import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import random

# 固定随机种子，确保每次采样的结果完全一致
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# ==============================================================================
# 1. 配置区域
# ==============================================================================

REAL_H5_PATH = r'/root/autodl-tmp/DataSet/ScanObjectNN_h5_files/h5_files/main_split/test_objectdataset.h5'
PHYSICS_DIR = r'/root/autodl-tmp/DataSet/shapenetcore_partanno_segmentation_benchmark_v0_physics_level_4_ours_full'
GAUSSIAN_DIR = r'/root/autodl-tmp/DataSet/shapenetcore_partanno_segmentation_benchmark_v0_gaussian_level_2'

MAX_SAMPLES = 200
OUTPUT_DIR = './validation_results'


# ==============================================================================
# 2. 核心计算函数 (保持不变)
# ==============================================================================

def compute_local_plane_residuals(points, k=20):
    if len(points) < k: return np.array([])
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    residuals = []
    step = 5
    for i in range(0, len(points), step):
        local_cloud = points[idx[i]]
        centroid = np.mean(local_cloud, axis=0)
        centered = local_cloud - centroid
        try:
            u, s, vh = np.linalg.svd(centered)
            normal = vh[2, :]
            dist = np.abs(np.dot(points[i] - centroid, normal))
            residuals.append(dist)
        except:
            pass
    return np.array(residuals)


def load_scanobjectnn(h5_path, max_samples):
    print(f"[Real] 正在读取: {h5_path}")
    all_points = []
    try:
        with h5py.File(h5_path, 'r') as f:
            key = next((k for k in ['data', 'points', 'tr_cloud'] if k in f.keys()), None)
            if key is None: raise KeyError(f"H5 keys not found.")
            data = f[key][:]
            if data.shape[0] > max_samples:
                indices = np.random.choice(data.shape[0], max_samples, replace=False)
                data = data[indices]
            for item in data:
                all_points.append(item[:, :3])
    except Exception as e:
        print(f"Error: {e}")
        return []
    return all_points


def load_simulated_folder(folder_path, max_samples, label="Sim"):
    print(f"[{label}] 正在扫描文件夹: {folder_path}")
    files = sorted(glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True))
    if not files: return []
    if len(files) > max_samples:
        files = np.random.choice(files, max_samples, replace=False)
    loaded_pcds = []
    for f in tqdm(files, desc=f"Loading {label}"):
        try:
            pts = np.loadtxt(f)[:, :3]
            loaded_pcds.append(pts)
        except:
            pass
    return loaded_pcds


def aggregate_residuals(pcd_list, desc="Processing"):
    total_residuals = []
    for pcd in tqdm(pcd_list, desc=desc):
        res = compute_local_plane_residuals(pcd, k=20)
        total_residuals.extend(res)
    return np.array(total_residuals)


# ==============================================================================
# 3. 新增：高级指标计算
# ==============================================================================

def compute_metrics(dist_p, dist_q, bins):
    """
    计算 P (Target/Real) 和 Q (Sim) 之间的多维度指标
    """
    # 1. 概率分布 (PDF)
    p_hist, _ = np.histogram(dist_p, bins=bins, density=True)
    q_hist, _ = np.histogram(dist_q, bins=bins, density=True)

    # 避免除零
    epsilon = 1e-10
    p_hist += epsilon
    q_hist += epsilon

    # 2. KL 散度 (Kullback-Leibler) - 越低越好
    kl = stats.entropy(p_hist, q_hist)

    # 3. JS 散度 (Jensen-Shannon) - 越低越好 (0~1)
    js = jensenshannon(p_hist, q_hist)

    # 4. Wasserstein 距离 (Earth Mover's Distance) - 越低越好
    # 不需要 binning，直接用原始数据计算
    # 为了计算速度，如果数据量太大，可以随机下采样
    limit_ws = 10000
    p_sample = np.random.choice(dist_p, min(len(dist_p), limit_ws))
    q_sample = np.random.choice(dist_q, min(len(dist_q), limit_ws))
    ws_dist = stats.wasserstein_distance(p_sample, q_sample)

    # 5. 统计矩对比 (Kurtosis 峰度)
    # 真实 LiDAR 噪声通常是 Heavy-tailed (高 Kurtosis)
    # Gaussian 噪声 Kurtosis ≈ 0 (Fisher definition)
    kurt_p = stats.kurtosis(dist_p)
    kurt_q = stats.kurtosis(dist_q)
    kurt_diff = abs(kurt_p - kurt_q)

    return {
        "KL": kl,
        "JS": js,
        "Wasserstein": ws_dist,
        "Kurtosis": kurt_q,  # 记录模拟数据的峰度
        "Kurtosis_Ref": kurt_p  # 记录真实数据的峰度
    }


# ==============================================================================
# 4. 主程序
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    real_pcds = load_scanobjectnn(REAL_H5_PATH, MAX_SAMPLES)
    phys_pcds = load_simulated_folder(PHYSICS_DIR, MAX_SAMPLES, label="Physics")
    gauss_pcds = load_simulated_folder(GAUSSIAN_DIR, MAX_SAMPLES, label="Gaussian")

    if not real_pcds or not phys_pcds or not gauss_pcds:
        print("Dataset loading failed.")
        return

    # 2. 计算残差
    print("\n>>> Computing Residuals...")
    res_real = aggregate_residuals(real_pcds, desc="Real")
    res_phys = aggregate_residuals(phys_pcds, desc="Physics")
    res_gauss = aggregate_residuals(gauss_pcds, desc="Gaussian")

    # 3. 准备统计
    limit = np.percentile(res_real, 99.5)  # 稍微扩大一点范围看长尾
    bins = np.linspace(0, limit, 60)

    # 4. 计算指标
    print("\n>>> Calculating Metrics...")
    metrics_phys = compute_metrics(res_real, res_phys, bins)
    metrics_gauss = compute_metrics(res_real, res_gauss, bins)

    # 打印结果表
    print("\n" + "=" * 65)
    print(f"{'Metric':<20} | {'Ours (Physics)':<20} | {'Baseline (Gaussian)':<20}")
    print("-" * 65)
    print(
        f"{'KL Divergence':<20} | {metrics_phys['KL']:.4f} {'(Best)' if metrics_phys['KL'] < metrics_gauss['KL'] else '' :<8} | {metrics_gauss['KL']:.4f}")
    print(
        f"{'JS Divergence':<20} | {metrics_phys['JS']:.4f} {'(Best)' if metrics_phys['JS'] < metrics_gauss['JS'] else '' :<8} | {metrics_gauss['JS']:.4f}")
    print(
        f"{'Wasserstein Dist':<20} | {metrics_phys['Wasserstein']:.4f} {'(Best)' if metrics_phys['Wasserstein'] < metrics_gauss['Wasserstein'] else '' :<8} | {metrics_gauss['Wasserstein']:.4f}")
    print("-" * 65)
    print(
        f"{'Kurtosis (Peak)':<20} | {metrics_phys['Kurtosis']:.2f} (Ref: {metrics_phys['Kurtosis_Ref']:.2f}) | {metrics_gauss['Kurtosis']:.2f}")
    print("=" * 65)
    print("Note: Kurtosis of Gaussian dist is ~0. Real LiDAR noise is typically >0 (Heavy-tailed).")

    # ==============================================================================
    # 5. 绘图 (拆分为独立的 PDF 和 CDF 图)
    # ==============================================================================

    # --- 图 1: PDF Comparison (Log Scale) ---
    plt.figure(figsize=(9, 6))
    x_axis = 0.5 * (bins[1:] + bins[:-1])

    def get_hist(data):
        h, _ = np.histogram(data, bins=bins, density=True)
        return h + 1e-10

    plt.semilogy(x_axis, get_hist(res_real), 'k-', lw=2.5, label='Real (ScanObjectNN)')
    plt.semilogy(x_axis, get_hist(res_phys), 'r-', lw=2, label=f'Ours (KL={metrics_phys["KL"]:.2f})')
    plt.semilogy(x_axis, get_hist(res_gauss), 'b--', lw=2, label=f'Baseline (KL={metrics_gauss["KL"]:.2f})')

    plt.title("Noise Distribution: PDF Comparison (Log Scale)", fontsize=14)
    plt.xlabel("Local Plane Residual (m)", fontsize=12)
    plt.ylabel("Log Probability Density", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    pdf_save_path = os.path.join(OUTPUT_DIR, 'fidelity_pdf_log.png')
    plt.tight_layout()
    plt.savefig(pdf_save_path, dpi=300)
    print(f"PDF Plot saved to {pdf_save_path}")
    plt.close()  # 关闭当前画布

    # --- 图 2: CDF Comparison (Tail Behavior) ---
    plt.figure(figsize=(9, 6))

    def plot_cdf(data, color, label, style='-'):
        sorted_data = np.sort(data)
        # 截断到显示范围以便对比
        sorted_data = sorted_data[sorted_data < limit]
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, color=color, linestyle=style, lw=2, label=label)

    plot_cdf(res_real, 'k', 'Real')
    plot_cdf(res_phys, 'r', f'Ours (WD={metrics_phys["Wasserstein"]:.3f})')
    plot_cdf(res_gauss, 'b', f'Baseline (WD={metrics_gauss["Wasserstein"]:.3f})', '--')

    plt.title("Noise Distribution: CDF Comparison", fontsize=14)
    plt.xlabel("Local Plane Residual (m)", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 如果你想重点展示长尾部分（例如 90% 以上的区域），可以取消下面这一行的注释
    # plt.ylim(0.85, 1.01)

    cdf_save_path = os.path.join(OUTPUT_DIR, 'fidelity_cdf.png')
    plt.tight_layout()
    plt.savefig(cdf_save_path, dpi=300)
    print(f"CDF Plot saved to {cdf_save_path}")
    plt.close()


if __name__ == "__main__":
    main()