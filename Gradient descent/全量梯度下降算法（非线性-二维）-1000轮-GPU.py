import matplotlib  # 导入 Matplotlib 主库，用于设置绘图后端和全局参数
import torch  # 导入 PyTorch，用于张量计算与自动求导（支持 GPU）
import matplotlib.pyplot as plt  # 导入 pyplot 接口，用于绘制二维图像
from datetime import datetime  # 导入 datetime，用于生成时间戳
import time  # 导入 time，用于计算程序运行时间
import os  # 导入 os，用于文件和目录操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 设置 Matplotlib 图形后端，适合桌面环境显示
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置默认字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = [  # 设置可用的中文字体列表
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示异常问题

# ============================
# 1. 设备选择（GPU 优先）
# ============================
use_cuda = torch.cuda.is_available()  # 判断是否支持 CUDA
device = torch.device("cuda" if use_cuda else "cpu")  # 选择计算设备

print("=" * 80)
print("计算设备信息")
print("=" * 80)
print(f"是否支持 CUDA: {use_cuda}")  # 输出是否支持 CUDA
print(f"当前计算设备: {device}")  # 输出当前使用的设备

if use_cuda:
    gpu_id = torch.cuda.current_device()  # 获取当前 GPU 编号
    gpu_name = torch.cuda.get_device_name(gpu_id)  # 获取 GPU 名称
    gpu_capability = torch.cuda.get_device_capability(gpu_id)  # 获取计算能力
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # 显存总量（GB）
    cuda_version = torch.version.cuda  # CUDA 运行时版本

    print(f"GPU 编号: {gpu_id}")
    print(f"GPU 名称: {gpu_name}")
    print(f"GPU 计算能力 (Compute Capability): {gpu_capability}")
    print(f"GPU 显存总量: {total_memory:.2f} GB")
    print(f"CUDA 版本: {cuda_version}")

print("=" * 80)

# ============================
# 2. 输出目录
# ============================
out_dir = "figures"  # 定义图像输出目录名称
os.makedirs(out_dir, exist_ok=True)  # 如果目录不存在则创建，存在则不报错
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳，用于文件命名

# ============================
# 3. 构造二维非线性样本数据（GPU）
# ============================
torch.manual_seed(42)  # 固定随机种子，保证实验结果可复现

n = 50  # 设置样本数量
X = torch.linspace(0, 2 * torch.pi, n, device=device)  # 在 [0, 2π] 区间生成输入样本，并放到 GPU/CPU
noise = torch.randn(n, device=device) * 0.2  # 生成均值为 0、标准差为 0.2 的高斯噪声
y = torch.sin(X) + noise  # 构造真实输出：正弦函数加噪声

print("真实数据规律：y = sin(x) + noise")  # 输出真实数据生成规律说明
print(f"样本数量: {n}")  # 输出样本数量信息

# ============================
# 4. 初始化模型参数（GPU）
# y_hat = w1*sin(x) + w2*cos(x) + b
# ============================
w1 = torch.zeros(1, device=device, requires_grad=True)  # 初始化 sin(x) 项权重，并开启梯度计算
w2 = torch.zeros(1, device=device, requires_grad=True)  # 初始化 cos(x) 项权重，并开启梯度计算
b  = torch.zeros(1, device=device, requires_grad=True)  # 初始化偏置项，并开启梯度计算

lr = 0.005  # 设置学习率（learning rate）
epochs = 1000  # 设置训练轮数（epoch 数）

loss_history = []  # 用于保存每一轮训练的损失值

# ============================
# 5. 训练开始计时（同步）
# ============================
if device.type == "cuda":  # 如果使用 GPU
    torch.cuda.synchronize()  # 强制同步 GPU，保证计时准确
start_time = time.time()  # 记录训练开始时间

print("=" * 80)  # 打印分隔线
print("开始全量梯度下降训练（正弦非线性回归，1000 轮）")  # 输出训练开始提示
print("=" * 80)  # 打印分隔线

# ============================
# 6. 全量梯度下降训练
# ============================
for epoch in range(epochs):  # 遍历训练轮数
    # 前向传播
    y_pred = w1 * torch.sin(X) + w2 * torch.cos(X) + b  # 根据当前参数计算模型预测值
    # MSE 损失
    loss = torch.mean((y_pred - y) ** 2)  # 计算均方误差（MSE）作为损失函数
    # 反向传播
    loss.backward()  # 自动计算损失对参数的梯度
    # 参数更新
    with torch.no_grad():  # 在不记录梯度的情况下更新参数
        w1 -= lr * w1.grad  # 根据梯度下降规则更新 w1
        w2 -= lr * w2.grad  # 根据梯度下降规则更新 w2
        b  -= lr * b.grad  # 根据梯度下降规则更新 b
    # 梯度清零
    w1.grad.zero_()  # 将 w1 的梯度清零，防止梯度累积
    w2.grad.zero_()  # 将 w2 的梯度清零
    b.grad.zero_()  # 将 b 的梯度清零

    loss_history.append(loss.item())  # 保存当前轮的损失值（转为 Python 标量）
    # 控制台输出（每 100 轮一次）
    if (epoch + 1) % 100 == 0 or epoch == 0:  # 控制打印频率
        print(  # 打印当前训练状态
            f"第 {epoch+1:04d} 轮 | "
            f"Loss={loss.item():.8f} | "
            f"w1={w1.item():.6f}, w2={w2.item():.6f}, b={b.item():.6f}"
        )

# ============================
# 7. 训练结束计时（同步）
# ============================
if device.type == "cuda":  # 如果使用 GPU
    torch.cuda.synchronize()  # 再次同步 GPU，保证计时结束准确
end_time = time.time()  # 记录训练结束时间

elapsed_time = end_time - start_time  # 计算总训练耗时

print("=" * 80)  # 打印分隔线
print("训练结束")  # 输出训练结束提示
print(f"总训练时间: {elapsed_time:.6f} 秒")  # 输出训练总耗时
print("=" * 80)  # 打印分隔线

# ============================
# 8. 拟合结果可视化（转回 CPU）
# ============================
X_cpu = X.detach().cpu()  # 将输入数据从 GPU 分离并拷贝到 CPU
y_cpu = y.detach().cpu()  # 将真实标签从 GPU 分离并拷贝到 CPU

y_fit = (w1 * torch.sin(X) + w2 * torch.cos(X) + b).detach().cpu()  # 计算最终预测结果并转到 CPU

plt.figure(figsize=(7, 5))  # 创建绘图窗口并设置大小
plt.scatter(X_cpu, y_cpu, label="样本数据", alpha=0.7)  # 绘制原始样本散点图
plt.plot(X_cpu, y_fit, color="black", linewidth=2, label="最终拟合模型")  # 绘制最终拟合曲线
plt.xlabel("输入 x")  # 设置 x 轴标签
plt.ylabel("输出 y")  # 设置 y 轴标签
plt.title("全量梯度下降最终结果（正弦非线性回归，1000 轮）")  # 设置图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

fit_path = f"{out_dir}/gpu_sine_fit_1000epochs_{timestamp}.png"  # 构造拟合结果图像保存路径
plt.savefig(fit_path, dpi=600)  # 保存拟合结果图像
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口

# ============================
# 9. 损失下降曲线
# ============================
plt.figure(figsize=(6, 4))  # 创建损失曲线绘图窗口
plt.plot(loss_history, linewidth=2)  # 绘制损失随训练轮数变化曲线
plt.xlabel("训练轮数（Epoch）")  # 设置 x 轴标签
plt.ylabel("均方误差（MSE）")  # 设置 y 轴标签
plt.title("全量梯度下降损失曲线（1000 轮）")  # 设置图像标题
plt.grid(True)  # 显示网格

loss_path = f"{out_dir}/gpu_sine_loss_1000epochs_{timestamp}.png"  # 构造损失曲线图像保存路径
plt.savefig(loss_path, dpi=600)  # 保存损失曲线图像
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口
