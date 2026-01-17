import matplotlib  # Matplotlib 主库
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
from mpl_toolkits.mplot3d import Axes3D  # 三维绘图支持
from datetime import datetime  # 时间戳
import os  # 文件操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 桌面后端
matplotlib.rcParams['font.family'] = 'sans-serif'  # 无衬线字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ============================================================
# 1. 输出目录
# ============================================================
out_dir = "figures"  # 图片保存目录
os.makedirs(out_dir, exist_ok=True)  # 创建目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳

# ============================================================
# 2. 构造数据
# ============================================================
np.random.seed(0)  # 固定随机种子
N = 120  # 样本数
x = np.linspace(-2, 2, N)  # 输入区间
noise = 0.3 * np.random.randn(N)  # 噪声
y = x**3 - x + noise  # 非线性真实函数

# ============================================================
# 3. 非线性回归模型
# ============================================================
def model(x, w1, w2):  # 模型定义
    return w1 * x**3 + w2 * x  # 预测输出

# ============================================================
# 4. 非对称非凸损失函数（w1 与 w2 同时正则）
# ============================================================
alpha1 = 0.15  # w1 双井强度
a1 = 2.0       # w1 谷底间距
beta1 = 0.08   # w1^3 非对称
gamma1 = -0.3  # w1 线性倾斜

alpha2 = 0.12  # w2 双井强度
a2 = 1.5       # w2 谷底间距
beta2 = 0.06   # w2^3 非对称
gamma2 = 0.2   # w2 线性倾斜

def loss(w1, w2):  # 总损失
    y_hat = model(x, w1, w2)  # 预测
    mse = np.mean((y - y_hat)**2)  # 数据拟合项
    reg_w1 = (  # w1 正则
        alpha1 * (w1**2 - a1**2)**2
        + beta1 * w1**3
        + gamma1 * w1
    )
    reg_w2 = (  # w2 正则（新增）
        alpha2 * (w2**2 - a2**2)**2
        + beta2 * w2**3
        + gamma2 * w2
    )
    return mse + reg_w1 + reg_w2  # 总损失

# ============================================================
# 5. 梯度计算（全量梯度）
# ============================================================
def gradients(w1, w2):  # 梯度
    y_hat = model(x, w1, w2)  # 预测
    error = y_hat - y  # 残差

    # 数据项梯度
    dw1_data = np.mean(2 * error * x**3)
    dw2_data = np.mean(2 * error * x)

    # 正则项梯度
    dw1_reg = (
        4 * alpha1 * w1 * (w1**2 - a1**2)
        + 3 * beta1 * w1**2
        + gamma1
    )
    dw2_reg = (
        4 * alpha2 * w2 * (w2**2 - a2**2)
        + 3 * beta2 * w2**2
        + gamma2
    )

    dw1 = dw1_data + dw1_reg  # 合并 w1 梯度
    dw2 = dw2_data + dw2_reg  # 合并 w2 梯度
    return dw1, dw2

# ============================================================
# 6. 全量梯度下降（记录轨迹）
# ============================================================
def gradient_descent(w1_init, w2_init, lr=0.006, epochs=600):
    w1, w2 = w1_init, w2_init  # 初始化
    traj = []  # 轨迹

    for epoch in range(epochs):
        l = loss(w1, w2)  # 当前损失
        dw1, dw2 = gradients(w1, w2)  # 梯度
        w1 -= lr * dw1  # 更新
        w2 -= lr * dw2
        traj.append([w1, w2, l])  # 记录
        print(f"第 {epoch+1:03d} 轮 | w1={w1:.4f} w2={w2:.4f} | 损失={l:.4f}")

    return np.array(traj)

# ============================================================
# 7. 不同初始化（更明显的分化）
# ============================================================
traj_A = gradient_descent(-3.0, -2.0)  # 初始化 A
traj_B = gradient_descent(3.0, 2.0)    # 初始化 B

# ============================================================
# 8. 数据分布 + 拟合曲线
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="原始数据", alpha=0.6)

y_fit_A = model(x, traj_A[-1, 0], traj_A[-1, 1])
y_fit_B = model(x, traj_B[-1, 0], traj_B[-1, 1])

plt.plot(x, y_fit_A, label="拟合结果（初始化 A）", linewidth=2)
plt.plot(x, y_fit_B, label="拟合结果（初始化 B）", linewidth=2)

plt.xlabel("输入 x")
plt.ylabel("输出 y")
plt.title("正则项同时作用于 w1 与 w2 时的拟合结果")
plt.legend()
plt.grid(True)

plt.savefig(f"{out_dir}/对比拟合曲线_w1_w2_{timestamp}.png", dpi=600)
plt.show()

# ============================================================
# 9. 三维损失曲面 + 梯度下降轨迹
# ============================================================
w1_vals = np.linspace(-4, 4, 160)
w2_vals = np.linspace(-4, 4, 160)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
L = np.vectorize(loss)(W1, W2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(W1, W2, L, cmap="viridis", alpha=0.65)
ax.plot(traj_A[:, 0], traj_A[:, 1], traj_A[:, 2],
        color="red", linewidth=3, label="梯度轨迹（初始化 A）")
ax.plot(traj_B[:, 0], traj_B[:, 1], traj_B[:, 2],
        color="orange", linewidth=3, label="梯度轨迹（初始化 B）")

ax.set_xlabel("参数 w1")
ax.set_ylabel("参数 w2")
ax.set_zlabel("损失值")
ax.set_title("w1 与 w2 同时正则时的非对称非凸损失景观")
ax.legend()

plt.savefig(f"{out_dir}/三维损失曲面_w1_w2_{timestamp}.png", dpi=600)
plt.show()
