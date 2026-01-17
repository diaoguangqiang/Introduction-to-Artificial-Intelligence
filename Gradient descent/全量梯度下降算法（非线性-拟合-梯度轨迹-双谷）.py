import matplotlib  # Matplotlib 主库
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
from mpl_toolkits.mplot3d import Axes3D  # 三维绘图支持
from datetime import datetime  # 时间戳
import os  # 文件操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 设置 Matplotlib 图形后端，适合桌面环境
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置默认字体族
matplotlib.rcParams['font.sans-serif'] = [  # 设置可用的中文字体
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ============================================================
# 1. 输出目录
# ============================================================
out_dir = "figures"  # 图片保存目录
os.makedirs(out_dir, exist_ok=True)  # 若目录不存在则创建
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳

# ============================================================
# 2. 构造对称数据分布
# ============================================================
np.random.seed(0)  # 固定随机种子，保证可复现
N = 120  # 样本数量
x = np.linspace(-2, 2, N)  # 对称输入区间
noise = 0.3 * np.random.randn(N)  # 高斯噪声
y = x**3 - x + noise  # 非线性真实函数（奇函数）

# ============================================================
# 3. 非线性模型
# ============================================================
def model(x, w1, w2):  # 定义非线性回归模型
    return w1 * x**3 + w2 * x  # 模型预测输出

# ============================================================
# 4. 非对称损失函数（核心）
# ============================================================
alpha = 0.15  # 双井势强度
a = 2.0       # 谷底位置间距
beta = 0.08   # w1^3 非对称项系数
gamma = -0.3  # 线性倾斜项系数

def loss(w1, w2):  # 定义总损失函数
    y_hat = model(x, w1, w2)  # 模型预测值
    mse = np.mean((y - y_hat)**2)  # 均方误差项
    reg = (  # 非对称正则项
        alpha * (w1**2 - a**2)**2  # 对称双井势
        + beta * w1**3             # 破坏左右对称
        + gamma * w1               # 整体倾斜
    )
    return mse + reg  # 返回总损失

# ============================================================
# 5. 梯度计算（全量梯度）
# ============================================================
def gradients(w1, w2):  # 计算损失函数梯度
    y_hat = model(x, w1, w2)  # 模型预测
    error = y_hat - y  # 残差

    dw1_data = np.mean(2 * error * x**3)  # 数据项对 w1 的梯度
    dw2 = np.mean(2 * error * x)  # 数据项对 w2 的梯度

    dw1_reg = (  # 正则项对 w1 的梯度
        4 * alpha * w1 * (w1**2 - a**2)  # 双井势导数
        + 3 * beta * w1**2               # w1^3 项导数
        + gamma                           # 线性项导数
    )

    dw1 = dw1_data + dw1_reg  # 合并 w1 总梯度
    return dw1, dw2  # 返回梯度

# ============================================================
# 6. 全量梯度下降（记录轨迹）
# ============================================================
def gradient_descent(w1_init, w2_init, lr=0.008, epochs=500):  # 全量梯度下降
    w1, w2 = w1_init, w2_init  # 参数初始化
    trajectory = []  # 记录参数与损失轨迹

    for epoch in range(epochs):  # 训练循环
        l = loss(w1, w2)  # 当前损失
        dw1, dw2 = gradients(w1, w2)  # 计算梯度
        w1 -= lr * dw1  # 更新 w1
        w2 -= lr * dw2  # 更新 w2
        trajectory.append([w1, w2, l])  # 保存当前轨迹
        print(f"第 {epoch+1:03d} 轮 | w1={w1:.4f} w2={w2:.4f} | 损失={l:.4f}")  # 输出训练过程

    return np.array(trajectory)  # 返回轨迹数组

# ============================================================
# 7. 两个不同初始化（落入不同谷底）
# ============================================================
traj_left = gradient_descent(-3.0, 0.5)   # 左侧初始化
traj_right = gradient_descent(3.0, -0.5)  # 右侧初始化

# ============================================================
# 8. 数据分布 + 拟合曲线
# ============================================================
plt.figure(figsize=(8, 6))  # 创建画布
plt.scatter(x, y, label="原始数据", alpha=0.6)  # 绘制数据点

y_fit_left = model(x, traj_left[-1, 0], traj_left[-1, 1])  # 左初始化拟合结果
y_fit_right = model(x, traj_right[-1, 0], traj_right[-1, 1])  # 右初始化拟合结果

plt.plot(x, y_fit_left, label="拟合曲线（左初始化）", linewidth=2)  # 拟合曲线
plt.plot(x, y_fit_right, label="拟合曲线（右初始化）", linewidth=2)  # 拟合曲线

plt.xlabel("输入 x")  # x 轴标签
plt.ylabel("输出 y")  # y 轴标签
plt.title("数据分布与非线性拟合结果")  # 图标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

plt.savefig(f"{out_dir}/拟合曲线_{timestamp}.png", dpi=600)  # 保存图片
plt.show()  # 显示图片

# ============================================================
# 9. 三维损失曲面 + 梯度下降轨迹
# ============================================================
w1_vals = np.linspace(-4, 4, 160)  # w1 范围
w2_vals = np.linspace(-3, 3, 120)  # w2 范围
W1, W2 = np.meshgrid(w1_vals, w2_vals)  # 构建参数网格
L = np.vectorize(loss)(W1, W2)  # 计算损失曲面

fig = plt.figure(figsize=(10, 8))  # 创建画布
ax = fig.add_subplot(111, projection="3d")  # 三维坐标系

ax.plot_surface(W1, W2, L, cmap="viridis", alpha=0.7)  # 绘制损失曲面
ax.plot(traj_left[:, 0], traj_left[:, 1], traj_left[:, 2],
        color="red", linewidth=3, label="梯度轨迹（左初始化）")  # 梯度轨迹
ax.plot(traj_right[:, 0], traj_right[:, 1], traj_right[:, 2],
        color="orange", linewidth=3, label="梯度轨迹（右初始化）")  # 梯度轨迹

ax.set_xlabel("参数 w1")  # x 轴
ax.set_ylabel("参数 w2")  # y 轴
ax.set_zlabel("损失值")  # z 轴
ax.set_title("非对称损失曲面与梯度下降轨迹")  # 图标题
ax.legend()  # 显示图例

plt.savefig(f"{out_dir}/三维损失曲面_{timestamp}.png", dpi=600)  # 保存图片
plt.show()  # 显示图片
