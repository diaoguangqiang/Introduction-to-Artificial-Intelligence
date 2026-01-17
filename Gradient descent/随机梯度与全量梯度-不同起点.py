import matplotlib  # 导入 matplotlib 主库
import numpy as np  # 导入 numpy，用于数值计算
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图接口
from datetime import datetime  # 导入 datetime，用于生成时间戳
import os  # 导入 os，用于文件夹操作

matplotlib.use("TkAgg")  # 设置 Matplotlib 图形后端，适合桌面交互式显示
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置默认字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = [  # 设置可用的中文字体列表
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'  # 常见中文字体
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# =========================
# 0. 基本设置
# =========================
np.random.seed(42)  # 固定随机种子，保证实验结果可复现
os.makedirs("figures", exist_ok=True)  # 创建 figures 文件夹（若已存在则不报错）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳，用于文件命名

# =========================
# 1. 定义损失函数与梯度
# =========================
a, b = 4.0, 0.5   # 椭圆损失函数的参数，控制不同方向的曲率

def loss(w):  # 定义损失函数 L(w)
    return (w[0]**2)/a + (w[1]**2)/b  # 椭圆型二次凸函数

def grad(w):  # 定义损失函数的梯度
    return np.array([2*w[0]/a, 2*w[1]/b])  # 对 w1 和 w2 分别求偏导

# =========================
# 2. 在同一等高线上选两个不同起点
# =========================
C = 40.0  # 指定等高线的常数值（能量相同）
theta_gd = np.pi / 2.5  # 全量梯度下降（GD）的起点角度
theta_sgd = 3*np.pi/5  # 随机梯度下降（SGD）的起点角度

w0_gd = np.array([  # 计算 GD 的初始点坐标
    np.sqrt(a*C) * np.cos(theta_gd),  # w1 初始值（椭圆参数方程）
    np.sqrt(b*C) * np.sin(theta_gd)   # w2 初始值（椭圆参数方程）
])

w0_sgd = np.array([  # 计算 SGD 的初始点坐标
    np.sqrt(a*C) * np.cos(theta_sgd),  # w1 初始值
    np.sqrt(b*C) * np.sin(theta_sgd)   # w2 初始值
])

# =========================
# 3. 超参数
# =========================
lr = 0.1  # 学习率，控制每一步更新幅度
steps = 40  # 迭代步数（更新次数）

# =========================
# 4. 全量梯度下降（GD）
# =========================
w_gd = w0_gd.copy()  # 初始化 GD 的参数向量
traj_gd = [w_gd.copy()]  # 用列表记录 GD 的参数轨迹

for _ in range(steps):  # 进行多步迭代
    w_gd = w_gd - lr * grad(w_gd)  # 按全量梯度下降公式更新参数
    traj_gd.append(w_gd.copy())  # 保存当前参数到轨迹列表

traj_gd = np.array(traj_gd)  # 将 GD 轨迹列表转换为 numpy 数组

# =========================
# 5. 随机梯度下降（SGD）
# =========================
w_sgd = w0_sgd.copy()  # 初始化 SGD 的参数向量
traj_sgd = [w_sgd.copy()]  # 用列表记录 SGD 的参数轨迹

noise_std = 0.8  # 随机噪声的标准差，用于模拟随机梯度的不确定性

for _ in range(steps):  # 进行多步迭代
    noise = np.random.randn(2) * noise_std  # 生成二维高斯噪声
    w_sgd = w_sgd - lr * (grad(w_sgd) + noise)  # 梯度 + 噪声更新参数
    traj_sgd.append(w_sgd.copy())  # 保存当前参数到轨迹列表

traj_sgd = np.array(traj_sgd)  # 将 SGD 轨迹列表转换为 numpy 数组

# =========================
# 6. 绘制等高线
# =========================
x = np.linspace(-6, 6, 400)  # 在 w1 方向生成网格点
y = np.linspace(-6, 6, 400)  # 在 w2 方向生成网格点
X, Y = np.meshgrid(x, y)  # 生成二维网格
Z = (X**2)/a + (Y**2)/b  # 计算每个网格点对应的损失值

plt.figure(figsize=(7, 6))  # 创建绘图窗口
plt.contour(X, Y, Z, levels=12, colors="black", linewidths=1)  # 绘制椭圆等高线

# =========================
# 7. 绘制轨迹
# =========================
plt.plot(  # 绘制 GD 的参数更新轨迹
    traj_gd[:,0], traj_gd[:,1],
    color="blue", linewidth=2, marker="o",
    label="全量梯度下降"
)

plt.plot(  # 绘制 SGD 的参数更新轨迹
    traj_sgd[:,0], traj_sgd[:,1],
    color="purple", linewidth=2, marker="x",
    label="随机梯度下降"
)

plt.scatter(*w0_gd, color="blue", s=80)  # 标记 GD 的起点
plt.scatter(*w0_sgd, color="purple", s=80)  # 标记 SGD 的起点

plt.scatter(0, 0, marker="*", s=200, color="red")  # 标记全局最优点（原点）

plt.xlabel(r"$w_1$")  # 设置 x 轴标签
plt.ylabel(r"$w_2$")  # 设置 y 轴标签
plt.title("GD 与 SGD（不同起点）")  # 设置图标题
plt.legend()  # 显示图例
plt.axis("equal")  # 保证坐标轴比例一致，避免椭圆变形

# =========================
# 8. 保存图片
# =========================
plt.savefig(f"figures/gd_vs_sgd_same_contour_{timestamp}.png", dpi=600)  # 保存图像（带时间戳）
plt.show()  # 显示图像
plt.close()  # 关闭绘图窗口，释放内存
