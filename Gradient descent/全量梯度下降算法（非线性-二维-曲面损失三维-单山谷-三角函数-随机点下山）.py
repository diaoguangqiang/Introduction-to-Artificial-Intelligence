import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================
# 1. 设备选择
# ============================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("=" * 80)
print("计算设备信息")
print("=" * 80)
print(f"是否支持 CUDA: {use_cuda}")
print(f"当前计算设备: {device}")
print("=" * 80)

# ============================
# 2. 输出目录
# ============================
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================
# 3. 构造二维非线性数据
# ============================
torch.manual_seed(42)

n = 60
X = torch.linspace(0, 2 * torch.pi, n, device=device)
noise = torch.randn(n, device=device) * 0.2
y = torch.sin(X) + noise

print("真实数据规律：y = sin(x) + ε")
print(f"样本数量: {n}")

# ============================
# 4. 初始化模型参数
# y_hat = w1*sin(x) + w2*cos(x) + b
# ============================
#w1 = torch.zeros(1, device=device, requires_grad=True)
#w2 = torch.zeros(1, device=device, requires_grad=True)
#b  = torch.zeros(1, device=device, requires_grad=True)

w1 = torch.tensor([5.0], device=device, requires_grad=True)
w2 = torch.tensor([-5.0], device=device, requires_grad=True)
b  = torch.tensor([3.0], device=device, requires_grad=True)

lr = 0.01
epochs = 300

loss_history = []
param_history = []   # (w1, b, loss)

# ============================
# 5. 全量梯度下降训练
# ============================
print("=" * 80)
print("开始全量梯度下降训练（非线性回归）")
print("=" * 80)

start_time = time.time()

for epoch in range(epochs):
    y_pred = w1 * torch.sin(X) + w2 * torch.cos(X) + b
    loss = torch.mean((y_pred - y) ** 2)

    loss.backward()

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        b  -= lr * b.grad

    w1.grad.zero_()
    w2.grad.zero_()
    b.grad.zero_()

    loss_history.append(loss.item())
    param_history.append([w1.item(), b.item(), loss.item()])

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(
            f"第 {epoch+1:03d} 轮 | "
            f"Loss={loss.item():.6f} | "
            f"w1={w1.item():.4f}, w2={w2.item():.4f}, b={b.item():.4f}"
        )

print("=" * 80)
print(f"训练结束，用时 {time.time() - start_time:.4f} 秒")
print("=" * 80)

param_history = np.array(param_history)

# ============================================================
# 6️⃣ 二维：数据空间中的拟合结果（先画这个）
# ============================================================
X_cpu = X.detach().cpu().numpy()
y_cpu = y.detach().cpu().numpy()

y_fit = (
    w1.detach().cpu() * torch.sin(X).cpu()
    + w2.detach().cpu() * torch.cos(X).cpu()
    + b.detach().cpu()
)

plt.figure(figsize=(7, 5))
plt.scatter(X_cpu, y_cpu, alpha=0.7, label="样本数据")
plt.plot(X_cpu, y_fit, color="black", linewidth=2, label="最终拟合曲线")
plt.xlabel("输入 x")
plt.ylabel("输出 y")
plt.title("二维数据空间中的非线性回归拟合结果")
plt.legend()
plt.grid(True)

fit_path = f"{out_dir}/sine_fit_2d_{timestamp}.png"
plt.savefig(fit_path, dpi=600)
plt.show()
plt.close()

# ============================================================
# 7️⃣ 三维：损失曲面 + 梯度下降轨迹（再画这个）
# ============================================================
w2_fixed = w2.item()

w1_range = np.linspace(-1.0, 1.5, 120)
b_range  = np.linspace(-1.0, 1.0, 120)
W1, B = np.meshgrid(w1_range, b_range)

Loss_surface = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        y_hat = (
            W1[i, j] * np.sin(X_cpu)
            + w2_fixed * np.cos(X_cpu)
            + B[i, j]
        )
        Loss_surface[i, j] = np.mean((y_hat - y_cpu) ** 2)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(W1, B, Loss_surface, cmap="viridis", alpha=0.75)
ax.plot(
    param_history[:, 0],
    param_history[:, 1],
    param_history[:, 2],
    color="red",
    marker="o",
    linewidth=2,
    label="梯度下降轨迹"
)

ax.set_xlabel("参数 w1")
ax.set_ylabel("参数 b")
ax.set_zlabel("损失值（MSE）")
ax.set_title("三维损失曲面与梯度下降轨迹（参数空间）")
ax.legend()

surface_path = f"{out_dir}/sine_loss_surface_{timestamp}.png"
plt.savefig(surface_path, dpi=600)
plt.show()
plt.close()

print("\n图像已保存：")
print(fit_path)
print(surface_path)
