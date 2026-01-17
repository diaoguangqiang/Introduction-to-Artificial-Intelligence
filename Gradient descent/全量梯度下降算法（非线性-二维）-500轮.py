import matplotlib  # 导入 Matplotlib 主库，用于设置绘图后端和全局参数
import numpy as np  # 导入 NumPy，用于数值计算和数组操作
import matplotlib.pyplot as plt  # 导入 pyplot 接口，用于绘制二维图像
from datetime import datetime  # 导入 datetime，用于生成时间戳
import os  # 导入 os，用于文件和目录操作

#验证GPU
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

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
# 1. 输出目录
# ============================
out_dir = "figures"  # 定义图像输出目录名称
os.makedirs(out_dir, exist_ok=True)  # 如果目录不存在则创建，存在则不报错
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳，用于文件命名

# ============================
# 2. 构造二维非线性样本数据（正弦）
# ============================
np.random.seed(42)  # 固定随机种子，保证实验结果可复现

n = 50  # 设置样本数量
X = np.linspace(0, 2 * np.pi, n)  # 在 [0, 2π] 区间内均匀生成输入样本
noise = np.random.normal(0, 0.2, n)  # 生成均值为 0、标准差为 0.2 的高斯噪声
y = np.sin(X) + noise  # 构造真实输出：正弦函数加噪声

print("真实数据规律：y = sin(x) + noise")  # 输出真实数据生成规律说明
print(f"样本数量: {n}")  # 输出样本数量信息

# ============================
# 3. 初始化模型参数
# 模型：y_hat = w1*sin(x) + w2*cos(x) + b
# ============================
w1 = 0.0  # 初始化 sin(x) 项的权重参数
w2 = 0.0  # 初始化 cos(x) 项的权重参数
b  = 0.0  # 初始化偏置项

lr = 0.005  # 设置学习率（learning rate），控制参数更新步长
epochs = 500  # 设置训练轮数（epoch 数）

loss_history = []  # 用于保存每一轮训练的损失值
w1_history = []  # 用于保存每一轮 w1 参数
w2_history = []  # 用于保存每一轮 w2 参数
b_history  = []  # 用于保存每一轮 b 参数

# ============================
# 4. 全量梯度下降训练（500 轮）
# ============================
print("=" * 80)  # 打印分隔线
print("开始全量梯度下降训练（正弦非线性回归，500 轮）")  # 输出训练开始提示
print("=" * 80)  # 打印分隔线

for epoch in range(epochs):  # 遍历训练轮数
    y_pred = w1 * np.sin(X) + w2 * np.cos(X) + b  # 根据当前参数计算模型预测值
    error = y_pred - y  # 计算预测值与真实值之间的误差
    loss = np.mean(error ** 2)  # 计算均方误差（MSE）作为损失函数
    # ---- 梯度（全量）----
    dw1 = (2 / n) * np.sum(error * np.sin(X))  # 计算 w1 的梯度
    dw2 = (2 / n) * np.sum(error * np.cos(X))  # 计算 w2 的梯度
    db  = (2 / n) * np.sum(error)  # 计算 b 的梯度
    # ---- 记录历史 ----
    loss_history.append(loss)  # 保存当前轮的损失值
    w1_history.append(w1)  # 保存当前轮的 w1 参数
    w2_history.append(w2)  # 保存当前轮的 w2 参数
    b_history.append(b)  # 保存当前轮的 b 参数
    # ---- 参数更新 ----
    w1 -= lr * dw1  # 按梯度下降规则更新 w1
    w2 -= lr * dw2  # 按梯度下降规则更新 w2
    b  -= lr * db  # 按梯度下降规则更新 b
    # ---- 控制台输出（每 50 轮一次）----
    if (epoch + 1) % 50 == 0 or epoch == 0:  # 控制打印频率
        print(  # 打印当前训练状态
            f"第 {epoch+1:03d} 轮 | "
            f"Loss={loss:.8f} | "
            f"w1={w1:.6f}, w2={w2:.6f}, b={b:.6f}"
        )

print("=" * 80)  # 打印分隔线
print("训练结束（500 轮）")  # 输出训练结束提示
print("=" * 80)  # 打印分隔线

# ============================
# 5. 二维拟合结果可视化
# ============================
plt.figure(figsize=(7, 5))  # 创建绘图窗口并设置大小
plt.scatter(X, y, label="样本数据", alpha=0.7)  # 绘制原始样本散点图

y_final = w1 * np.sin(X) + w2 * np.cos(X) + b  # 使用最终参数计算模型预测结果
plt.plot(X, y_final, color="black", linewidth=2, label="最终拟合模型")  # 绘制最终拟合曲线

plt.xlabel("输入 x")  # 设置 x 轴标签
plt.ylabel("输出 y")  # 设置 y 轴标签
plt.title("全量梯度下降最终结果（正弦非线性回归，500 轮）")  # 设置图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

fit_path = f"{out_dir}/sine_fit_500epochs_{timestamp}.png"  # 构造拟合结果图像保存路径
plt.savefig(fit_path, dpi=600)  # 保存拟合结果图像
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口

# ============================
# 6. 损失下降曲线
# ============================
plt.figure(figsize=(6, 4))  # 创建损失曲线绘图窗口
plt.plot(loss_history, linewidth=2)  # 绘制损失随训练轮数变化曲线
plt.xlabel("训练轮数（Epoch）")  # 设置 x 轴标签
plt.ylabel("均方误差（MSE）")  # 设置 y 轴标签
plt.title("全量梯度下降的损失下降曲线（500 轮）")  # 设置图像标题
plt.grid(True)  # 显示网格

loss_path = f"{out_dir}/sine_loss_500epochs_{timestamp}.png"  # 构造损失曲线图像保存路径
plt.savefig(loss_path, dpi=600)  # 保存损失曲线图像
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口

# ============================
# 7. 最终参数
# ============================
print("最终模型参数（500 轮）：")  # 输出最终模型参数说明
print(f"w1 = {w1:.8f}")  # 输出最终 w1 参数值
print(f"w2 = {w2:.8f}")  # 输出最终 w2 参数值
print(f"b  = {b:.8f}")  # 输出最终 b 参数值
print("图像已保存：")  # 输出提示信息
print(fit_path)  # 输出拟合结果图像路径
print(loss_path)  # 输出损失曲线图像路径
