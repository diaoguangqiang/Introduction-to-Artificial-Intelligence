import matplotlib  # Matplotlib 主库，用于绘图与后端设置
import torch  # PyTorch 深度学习框架
import numpy as np  # NumPy，用于数值计算与数组操作
import matplotlib.pyplot as plt  # Matplotlib 绘图接口
from datetime import datetime  # 用于生成时间戳
import time  # 用于计时
import os  # 用于文件与目录操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 指定 Matplotlib 使用 TkAgg 后端
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = [  # 指定可用的中文字体列表
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# ============================
# 1. 设备选择
# ============================
use_cuda = torch.cuda.is_available()  # 判断当前环境是否支持 CUDA
device = torch.device("cuda" if use_cuda else "cpu")  # 根据是否支持 CUDA 选择计算设备

print("=" * 80)  # 输出分隔线
print("计算设备信息")  # 输出提示信息
print(f"是否支持 CUDA: {use_cuda}")  # 输出 CUDA 支持情况
print(f"当前计算设备: {device}")  # 输出当前使用的计算设备
print("=" * 80)  # 输出分隔线

# ============================
# 2. 输出目录
# ============================
out_dir = "figures"  # 设置图像输出目录
os.makedirs(out_dir, exist_ok=True)  # 若目录不存在则创建
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间戳字符串

# ============================
# 3. 构造二维非线性数据
# ============================
torch.manual_seed(42)  # 固定随机种子，保证实验可复现

n = 200  # 设置样本数量（样本量较大，结果更稳定）
X = torch.linspace(0, 2 * torch.pi, n, device=device)  # 在 [0, 2π] 区间生成均匀采样点
noise = torch.randn(n, device=device) * 0.2  # 生成高斯噪声
y = torch.sin(X) + noise  # 构造真实数据 y = sin(x) + ε

print("真实数据规律：y = sin(x) + ε")  # 输出真实数据生成规律
print(f"样本数量: {n}")  # 输出样本数量

X_np = X.detach().cpu().numpy()  # 将 X 从 GPU 张量转为 NumPy 数组
y_np = y.detach().cpu().numpy()  # 将 y 从 GPU 张量转为 NumPy 数组

# ============================
# 4. 🔥 先扫描损失曲面，寻找“山顶”
# 固定 w2，只在 (w1, b) 平面上观察
# ============================
w2_fixed = 0.0  # 固定参数 w2 的取值

w1_range = np.linspace(-6, 6, 200)  # 设置 w1 的搜索范围
b_range  = np.linspace(-6, 6, 200)  # 设置 b 的搜索范围
W1, B = np.meshgrid(w1_range, b_range)  # 构造参数网格

Loss_surface = np.zeros_like(W1)  # 初始化损失曲面矩阵

for i in range(W1.shape[0]):  # 遍历 w1 网格
    for j in range(W1.shape[1]):  # 遍历 b 网格
        y_hat = (  # 计算当前参数下的预测值
            W1[i, j] * np.sin(X_np)  # w1 * sin(x)
            + w2_fixed * np.cos(X_np)  # w2 * cos(x)
            + B[i, j]  # 偏置项 b
        )
        Loss_surface[i, j] = np.mean((y_hat - y_np) ** 2)  # 计算均方误差损失

max_idx = np.unravel_index(np.argmax(Loss_surface), Loss_surface.shape)  # 找到损失最大的索引
w1_top = W1[max_idx]  # 对应的 w1 值
b_top  = B[max_idx]  # 对应的 b 值
max_loss = Loss_surface[max_idx]  # 最大损失值

print("🎯 找到损失曲面顶点（山顶）：")  # 输出提示信息
print(f"w1_top = {w1_top:.3f}, b_top = {b_top:.3f}, Loss = {max_loss:.3f}")  # 输出山顶参数

# ============================
# 5. 从“山顶”初始化模型参数
# ============================
w1 = torch.tensor([w1_top], device=device, requires_grad=True)  # 初始化 w1，并开启梯度
w2 = torch.tensor([w2_fixed], device=device, requires_grad=True)  # 初始化 w2，并开启梯度
b  = torch.tensor([b_top], device=device, requires_grad=True)  # 初始化 b，并开启梯度

lr = 0.01  # 设置学习率
epochs = 300  # 设置训练轮数

loss_history = []  # 用于记录每一轮的损失值
param_history = []  # 用于记录参数轨迹 (w1, b, loss)

# ============================
# 6. 全量梯度下降训练
# ============================
print("=" * 80)  # 输出分隔线
print("从损失曲面山顶开始梯度下降")  # 输出提示信息
print("=" * 80)  # 输出分隔线

start_time = time.time()  # 记录训练开始时间

for epoch in range(epochs):  # 训练循环
    y_pred = w1 * torch.sin(X) + w2 * torch.cos(X) + b  # 前向计算预测值
    loss = torch.mean((y_pred - y) ** 2)  # 计算均方误差损失

    loss.backward()  # 反向传播，计算梯度

    with torch.no_grad():  # 在无梯度模式下更新参数
        w1 -= lr * w1.grad  # 更新 w1
        w2 -= lr * w2.grad  # 更新 w2
        b  -= lr * b.grad  # 更新 b

    w1.grad.zero_()  # 清空 w1 的梯度
    w2.grad.zero_()  # 清空 w2 的梯度
    b.grad.zero_()  # 清空 b 的梯度

    loss_history.append(loss.item())  # 记录当前损失
    param_history.append([w1.item(), b.item(), loss.item()])  # 记录参数轨迹

    if (epoch + 1) % 50 == 0 or epoch == 0:  # 定期输出训练信息
        print(
            f"第 {epoch+1:03d} 轮 | "
            f"Loss={loss.item():.6f} | "
            f"w1={w1.item():.4f}, b={b.item():.4f}"
        )

print("=" * 80)  # 输出分隔线
print(f"训练结束，用时 {time.time() - start_time:.4f} 秒")  # 输出训练耗时
print("=" * 80)  # 输出分隔线

param_history = np.array(param_history)  # 将参数历史转换为 NumPy 数组

# ============================
# 7️⃣ 二维：数据空间中的拟合结果
# ============================
y_fit = (  # 计算最终拟合曲线
    w1.detach().cpu().numpy() * np.sin(X_np)
    + w2.detach().cpu().numpy() * np.cos(X_np)
    + b.detach().cpu().numpy()
)

plt.figure(figsize=(7, 5))  # 创建绘图窗口
plt.scatter(X_np, y_np, alpha=0.6, label="样本数据")  # 绘制散点图
plt.plot(X_np, y_fit, color="black", linewidth=2, label="最终拟合曲线")  # 绘制拟合曲线
plt.xlabel("输入 x")  # 设置 x 轴标签
plt.ylabel("输出 y")  # 设置 y 轴标签
plt.title("二维数据空间中的非线性回归拟合结果")  # 设置标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

fit_path = f"{out_dir}/fit_from_peak_{timestamp}.png"  # 拟合结果图像路径
plt.savefig(fit_path, dpi=300)  # 保存图像
plt.show()  # 显示图像
plt.close()  # 关闭绘图窗口

# ============================
# 8️⃣ 三维：损失曲面 + 梯度下降轨迹
# ============================
fig = plt.figure(figsize=(9, 7))  # 创建三维绘图窗口
ax = fig.add_subplot(111, projection="3d")  # 添加三维坐标轴

ax.plot_surface(W1, B, Loss_surface, cmap="viridis", alpha=0.75)  # 绘制损失曲面
ax.plot(
    param_history[:, 0],  # w1 轨迹
    param_history[:, 1],  # b 轨迹
    param_history[:, 2],  # 损失轨迹
    color="red",
    marker="o",
    linewidth=2,
    label="梯度下降轨迹"
)

ax.set_xlabel("参数 w1")  # 设置 x 轴标签
ax.set_ylabel("参数 b")  # 设置 y 轴标签
ax.set_zlabel("损失值（MSE）")  # 设置 z 轴标签
ax.set_title("从损失曲面山顶开始的梯度下降轨迹")  # 设置标题
ax.legend()  # 显示图例

surface_path = f"{out_dir}/loss_surface_with_peak_path_{timestamp}.png"  # 三维图像路径
plt.savefig(surface_path, dpi=300)  # 保存三维图像
plt.show()  # 显示图像
plt.close()  # 关闭绘图窗口

print("\n图像已保存：")  # 输出提示信息
print(fit_path)  # 输出二维拟合图路径
print(surface_path)  # 输出三维损失曲面图路径
