import os
import glob
import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

# ==========================================
# 1. 配置参数类
# ==========================================
class LiDARPhysicsParams:
    """
    存储 LiDAR 传感器的物理参数
    """

    def __init__(self, level=3):
        """
                初始化物理参数。
                level: 0 (无噪) -> 4 (极噪)
                """
        self.set_level(level)

    def set_level(self, level):
        self.level = level

        # 预设参数表
        # Level 0: 理想情况 (Ideal)
        if level == 0:
            self.beam_divergence = 1e-6  # 几乎无发散
            self.kappa = 0.0  # 无几何漂移
            self.saturation_offset = 0.0  # 无饱和偏移
            self.kinematic_drift = np.zeros(3)  # 无运动漂移
            self.env_noise_base = 0.0001  # 亚毫米级微扰
            self.correlation_scale = 1.0

        # Level 1: 高精度工业扫描 (High Precision)
        elif level == 1:
            self.beam_divergence = 0.0002  # 0.2 mrad (非常聚光)
            self.kappa = 0.001  # 极小漂移
            self.saturation_offset = 0.002  # 2mm
            self.kinematic_drift = np.array([0.001, 0.001, 0.0005])
            self.env_noise_base = 0.001  # 1mm 基础噪声
            self.correlation_scale = 10.0

        # Level 2: 标准车载雷达 (Standard Automotive) -> 推荐
        elif level == 2:
            self.beam_divergence = 0.0015  # 1.5 mrad (典型值)
            self.kappa = 0.003  # 适中漂移
            self.saturation_offset = 0.005  # 5mm
            self.kinematic_drift = np.array([0.003, 0.003, 0.001])
            self.env_noise_base = 0.003  # 3mm 基础噪声
            self.correlation_scale = 20.0

        # Level 3: 手持/低端设备 (Noisy/Handheld) -> 之前的版本
        elif level == 3:
            self.beam_divergence = 0.0035  # 3.5 mrad (光斑较大)
            self.kappa = 0.008  # 明显漂移
            self.saturation_offset = 0.015  # 1.5cm
            self.kinematic_drift = np.array([0.008, 0.008, 0.003])
            self.env_noise_base = 0.008  # 8mm 基础噪声
            self.correlation_scale = 40.0

        # Level 4: 恶劣天气/极限 (Severe/Adverse Weather)
        else:
            self.beam_divergence = 0.006  # 6 mrad (巨大光斑，飞点严重)
            self.kappa = 0.015  # 严重几何失真
            self.saturation_offset = 0.03  # 3cm
            self.kinematic_drift = np.array([0.015, 0.015, 0.005])
            self.env_noise_base = 0.015  # 1.5cm 基础噪声
            self.correlation_scale = 60.0

        # 通用常量
        self.grazing_threshold = np.deg2rad(75)
        self.tau_reflectivity = 0.8


# ==========================================
# 2. 物理感知模拟器核心 (Core Engine)
# ==========================================
class PhysicsAwareSimulator:
    def __init__(self, params: LiDARPhysicsParams):
        self.params = params

    def _compute_geometric_factors(self, points, normals, sensor_pos):
        """
        计算每个点的物理几何属性：距离 (Range) 和 入射角 (Incidence Angle)
        """
        # 计算视线向量
        vecs = points - sensor_pos
        ranges = np.linalg.norm(vecs, axis=1)  # [r]

        # 归一化视线向量
        # 添加微小量防止除零
        view_dirs = vecs / (ranges[:, np.newaxis] + 1e-8)

        # 计算点积 (View Dir * Normal)
        # 注意：法线可能指向物体内部或外部，取绝对值计算夹角
        dots = np.sum(view_dirs * normals, axis=1)
        dots = np.clip(dots, -1.0, 1.0)

        # theta: 光束与法线的夹角
        thetas = np.arccos(np.abs(dots))

        return ranges, thetas

    def _compute_systematic_bias(self, ranges, thetas, reflectivity):
        """
        计算系统偏差 b_i
        Bias = b_geo + b_rad + b_kin
        """
        # 1. 几何失真 (Geometric Distortion)
        # b_geo = kappa * r * tan(theta)
        # 限制 theta 防止数值不稳定
        clipped_theta = np.clip(thetas, 0, self.params.grazing_threshold)
        b_geo = self.params.kappa * ranges * np.tan(clipped_theta)

        # 2. 辐射失真 (Radiometric Distortion)
        # 模拟高反射率导致的波形饱和截断
        b_rad = np.zeros_like(ranges)
        saturated_mask = reflectivity > self.params.tau_reflectivity
        b_rad[saturated_mask] = self.params.saturation_offset

        # 3. 运动学失真 (Kinematic Distortion)
        # 这是一个矢量漂移，但在计算 magnitude 时我们在之后统一处理
        # 此处返回标量 magnitude 的一部分和矢量部分

        total_scalar_bias = b_geo + b_rad
        return total_scalar_bias, self.params.kinematic_drift

    def _compute_aleatoric_variance(self, ranges, thetas, reflectivity):
        """
        计算异方差 (Heteroscedastic Variance) sigma_i
        Sigma^2 = sigma_geo^2 + sigma_rad^2 + sigma_env^2
        """
        clipped_theta = np.clip(thetas, 0, self.params.grazing_threshold)

        # 1. 几何方差 (Geometric Variance)
        # 与光斑面积成正比: proportional to (r * tan(theta))^2
        sigma_geo_sq = (0.001 * ranges * np.tan(clipped_theta)) ** 2

        # 2. 辐射方差 (Radiometric Variance)
        # 与 SNR 成反比: proportional to r^2 / reflectivity
        sigma_rad_sq = (1e-5 * (ranges ** 2)) / (reflectivity + 1e-4)

        # 3. 环境方差 (Environmental Variance)
        sigma_env_sq = self.params.env_noise_base ** 2

        total_sigma = np.sqrt(sigma_geo_sq + sigma_rad_sq + sigma_env_sq)
        return total_sigma

    def _apply_grf(self, points, sigmas, ranges):
        """
        构建物理感知高斯随机场 (Physics-Aware GRF) 并采样噪声

        """
        N = len(points)
        if N == 0: return np.zeros((0, 3))

        # 1. 计算欧氏距离矩阵
        dists = cdist(points, points, metric='euclidean')

        # 2. 计算动态长度尺度 (Length Scale) l(r)
        # l = gamma * r * scaling_factor
        # 距离越远，光斑越大，噪声的空间相关性越强
        avg_ranges = (ranges[:, None] + ranges[None, :]) / 2.0
        length_scales = self.params.beam_divergence * avg_ranges * self.params.correlation_scale

        # 3. 构建核函数 (Covariance Kernel)
        # R(p_i, p_j) = exp( -d^2 / (2 * l^2) )
        correlation_kernel = np.exp(- (dists ** 2) / (2 * (length_scales ** 2) + 1e-8))

        # 4. 构建协方差矩阵 K
        # K_ij = sigma_i * sigma_j * R_ij
        K = np.outer(sigmas, sigmas) * correlation_kernel

        # 添加 jitter 保证正定性
        K += np.eye(N) * 1e-6

        # 5. Cholesky 分解采样
        try:
            L = np.linalg.cholesky(K)
            z = np.random.standard_normal((N, 3))
            # 噪声具有空间相关性
            correlated_noise = L @ z
        except np.linalg.LinAlgError:
            # 如果矩阵数值不稳定，回退到对角噪声
            warnings.warn("GRF Cholesky decomposition failed. Falling back to independent noise.")
            correlated_noise = np.random.normal(0, sigmas[:, None], (N, 3))

        return correlated_noise

    def simulate_scan(self, points, normals, sensor_pos):
        """
        对单次可见的扫描点进行加噪
        """
        if len(points) == 0:
            return None

        # 1. 计算物理参数
        ranges, thetas = self._compute_geometric_factors(points, normals, sensor_pos)

        # 2. 模拟反射率 (基于 Lambertian 假设: rho ~ cos(theta))
        reflectivity = np.abs(np.cos(thetas)) * 0.9 + 0.1

        # 3. 计算误差预算
        scalar_bias, vec_bias_kin = self._compute_systematic_bias(ranges, thetas, reflectivity)
        sigmas = self._compute_aleatoric_variance(ranges, thetas, reflectivity)

        # 4. 生成 GRF 噪声
        noise = self._apply_grf(points, sigmas, ranges)

        # 5. 应用所有误差
        # P_meas = P_true + Bias_geometric(沿着法线) + Bias_kinematic + Noise_GRF
        # 注意：几何漂移通常沿着视线或法线，此处简化为沿法线漂移
        bias_displacement = normals * scalar_bias[:, np.newaxis] + vec_bias_kin

        corrupted_points = points + bias_displacement + noise

        return corrupted_points


# ==========================================
# 2.5 高斯噪声模拟器 (Baseline Simulator)
# ==========================================
class GaussianSimulator:
    """
    基准对照组：仅添加标准高斯白噪声 (White Noise)，
    不考虑波束发散、入射角或反射率。
    """

    def __init__(self, std_dev=0.01):
        """
        :param std_dev: 噪声标准差 (米).
        """
        self.std_dev = std_dev

    def simulate_scan(self, points, normals, sensor_pos):
        """
        接口与 PhysicsAwareSimulator 保持一致
        """
        if len(points) == 0:
            return None

        # 生成独立的高斯白噪声 (I.I.D Gaussian)
        # shape: [N, 3]
        noise = np.random.normal(loc=0.0, scale=self.std_dev, size=points.shape)

        # 直接叠加: P_meas = P_true + Gaussian_Noise
        corrupted_points = points + noise

        return corrupted_points

# ==========================================
# 3. ShapeNet 适配器 (Input/Output Handler)
# ==========================================
class ShapeNetScanner:
    def __init__(self, mode='physics', level=2):
        """
        :param mode: 'physics' (物理感知) 或 'gaussian' (高斯基准)
        :param level: 噪声等级 (0-4)
        """
        self.mode = mode
        self.level = level

        if mode == 'physics':
            self.params = LiDARPhysicsParams(level=level)
            self.simulator = PhysicsAwareSimulator(self.params)
            # print(f"[Scanner] Mode: Physics-Aware | Level: {level}")

        elif mode == 'gaussian':
            # 定义高斯噪声的难度分级，与物理参数大致对齐以便对比
            # Level 0: Clean
            # Level 1: 2mm (High Precision)
            # Level 2: 5mm (Automotive)
            # Level 3: 10mm (Handheld)
            # Level 4: 20mm (Severe)
            gaussian_levels = {
                0: 0.000,
                1: 0.002,
                2: 0.005,
                3: 0.010,
                4: 0.020
            }
            std_dev = gaussian_levels.get(level, 0.005)
            self.simulator = GaussianSimulator(std_dev=std_dev)
            # print(f"[Scanner] Mode: Gaussian Jitter | Level: {level} (Std={std_dev*1000:.1f}mm)")

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _estimate_normals(self, pcd):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        return pcd

    def _resample_with_labels(self, points, labels, target_N):
        """
        同步重采样点云和标签
        """
        curr_N = len(points)
        if curr_N == 0:
            return np.zeros((target_N, 3)), np.zeros((target_N, labels.shape[1]))

        if curr_N > target_N:
            idx = np.random.choice(curr_N, target_N, replace=False)
        else:
            idx = np.random.choice(curr_N, target_N, replace=True)

        return points[idx], labels[idx]

    def scan(self, input_data, view_mode='surround'):
        """
        :param input_data: [N, 3+K] Numpy 数组 (前3列为XYZ，后续为Label)
        :return: [N, 3+K] 带有物理噪声且保留Label的点云
        """
        input_points_np = input_data[:, :3]
        input_labels_np = input_data[:, 3:]  # 提取标签部分

        # 1. 转换为 Open3D 对象并估计法线
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_points_np)
        pcd = self._estimate_normals(pcd)

        all_points = np.asarray(pcd.points)
        all_normals = np.asarray(pcd.normals)

        # 2. 定义传感器位置
        r = 2.5
        if view_mode == 'single_random':
            phi, theta = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
            sensor_positions = [
                np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])]
        else:
            sensor_positions = [
                np.array([r, 0, 0]), np.array([-r, 0, 0]),
                np.array([0, r, 0]), np.array([0, -r, 0]), np.array([0, 0, r])
            ]

        # 3. 执行多视角扫描
        scanned_points_list = []
        scanned_labels_list = []

        for pos in sensor_positions:
            _, pt_map = pcd.hidden_point_removal(pos, radius=1000)
            if len(pt_map) < 10: continue

            visible_points = all_points[pt_map]
            visible_normals = all_normals[pt_map]
            visible_labels = input_labels_np[pt_map]  # 同步提取标签

            # 物理加噪
            noisy_points = self.simulator.simulate_scan(visible_points, visible_normals, pos)

            if noisy_points is not None:
                scanned_points_list.append(noisy_points)
                scanned_labels_list.append(visible_labels)

        # 4. 融合
        if not scanned_points_list:
            return input_data

        final_points = np.vstack(scanned_points_list)
        final_labels = np.vstack(scanned_labels_list)

        # 5. 重采样回原始大小 (同步处理)
        resampled_points, resampled_labels = self._resample_with_labels(final_points, final_labels, len(input_data))

        # 合并返回 [N, 3+K]
        return np.hstack([resampled_points, resampled_labels])

def visualize_matplotlib(original_points, noisy_points):
    """
    使用 Matplotlib 绘制对比图
    """
    fig = plt.figure(figsize=(12, 6))

    # 1. 绘制原始点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c='gray', s=1, alpha=0.5, label='Original')
    ax1.set_title("Original Ideal CAD")
    ax1.set_axis_off() # 关闭坐标轴看起来更像 3D 模型

    # 2. 绘制噪声点云
    ax2 = fig.add_subplot(122, projection='3d')
    # 使用红色显示，模拟论文中的 Visualization
    ax2.scatter(noisy_points[:, 0], noisy_points[:, 1], noisy_points[:, 2],
                c='red', s=1, alpha=0.5, label='Physics Noise')
    ax2.set_title("Physics-Aware Simulation")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig("/root/autodl-tmp/DataSet/simulation_result.png", dpi=150) # 保存为图片
    print("可视化结果已保存为 'simulation_result.png'")
    # plt.show() # 如果在 Jupyter 中可以取消注释


# ==========================================
# 4. 批量处理逻辑 (新增与修改部分)
# ==========================================
def process_dataset(src_root, dst_root,noise_mode, noise_level):
    """
    增加了 noise_mode 参数
    """
    print(f"==================================================")
    print(f"模式: {noise_mode.upper()}") # 打印模式
    print(f"等级: Level {noise_level}")
    print(f"源: {src_root}")
    print(f"目标: {dst_root}")
    print(f"==================================================")

    # === 初始化对应的 Scanner ===
    scanner = ShapeNetScanner(mode=noise_mode, level=noise_level)

    # 1. 收集所有文件路径
    all_files = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith('.txt'):
                all_files.append(os.path.join(root, file))
    all_files = all_files[::10]  # 切片操作：从0开始，步长为10

    print(f"采样后共包含 {len(all_files)} 个 .txt 文件 (1/10)，准备开始处理...")

    # 2. 遍历处理
    success_count = 0
    fail_count = 0

    # 使用 tqdm 显示进度条
    for input_path in tqdm(all_files, desc=f"{noise_mode}-L{noise_level}"):
        try:
            # 计算相对路径，用于构建目标路径
            # 例如: src/02691156/xxx.txt -> 02691156/xxx.txt
            rel_path = os.path.relpath(input_path, src_root)

            # 构建目标文件的完整路径
            output_path = os.path.join(dst_root, rel_path)

            # 确保目标文件夹存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 读取数据
            # 假设数据格式为 [N, K] 纯文本
            input_data = np.loadtxt(input_path)

            # 执行物理加噪扫描
            # 如果文件过大，可以考虑先降采样，或者在 scanner 内部处理
            processed_data = scanner.scan(input_data, view_mode='surround')

            # 保存数据
            # fmt='%.6f' 保留6位小数，避免文件体积过大
            np.savetxt(output_path, processed_data, fmt='%.6f')

            success_count += 1

        except Exception as e:
            print(f"\n[Error] 处理文件失败: {input_path}")
            print(f"原因: {e}")
            fail_count += 1

    print(f"\n==================================================")
    print(f"处理完成 Summary:")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"结果保存在: {dst_root}")
    print(f"==================================================")


if __name__ == "__main__":

    # 1. 定义要运行的实验配置
    # 格式: (Mode, Level)
    experiments = [
        # ('physics', 2),  # 你的主实验数据 (Ours)
        ('gaussian', 2),  # 你的对比实验数据 (Baseline)
        # ('physics', 4), # 极端情况
    ]
    SRC_DIR = '/root/autodl-tmp/DataSet/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # 获取父目录
    parent_dir = os.path.dirname(SRC_DIR.rstrip('/'))
    base_folder_name = os.path.basename(SRC_DIR.rstrip('/'))

    for mode, level in experiments:
        # 2. 自动生成区分度高的文件夹名称
        # 例如: ..._physics_level_2 或 ..._gaussian_level_2

        # 移除原有的 'normal' 标记，替换为新的后缀
        clean_name = base_folder_name.replace("_normal", "")
        clean_name = clean_name.replace("normal", "")  # 防止名字不同

        new_folder_name = f"{clean_name}_{mode}_level_{level}"
        DST_DIR = os.path.join(parent_dir, new_folder_name)

        if not os.path.exists(SRC_DIR):
            print(f"错误: 源目录不存在 -> {SRC_DIR}")
            continue

        # 3. 运行处理
        process_dataset(SRC_DIR, DST_DIR, mode, level)

'''
# ==========================================
# 4. 运行示例
# ==========================================
if __name__ == "__main__":
    N_points = 2048
    file_path = '/root/autodl-tmp/DataSet/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1066b65c30d153e04c3a35cee92bb95b.txt'

    # 读取原始数据 [N, 6] 或 [N, 7] (假设包含 XYZ, Normal, Label)
    Points_Lable = np.loadtxt(file_path)

    # 假设我们要保留全部信息或特定列（例如前3列XYZ + 最后一列Label）
    # 这里我们取前2048个点的全部原始列
    dummy_input = Points_Lable[:N_points, :]

    scanner = ShapeNetScanner(level=2)

    print("--- 开始物理感知扫描模拟 (保留标签) ---")

    # 执行扫描
    output_data = scanner.scan(dummy_input, view_mode='surround')

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output_data.shape}")
    print(f"前5行标签示例:\n{output_data[:5, 3:]}")

    # 可视化部分 (保持不变，仅取前3列绘图)
    try:
        visualize_matplotlib(dummy_input[:, :3], output_data[:, :3])
    except Exception as e:
        print(f"可视化失败: {e}")
'''