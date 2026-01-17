import matplotlib  # 导入 Matplotlib 主库，用于设置绘图后端和全局参数
import numpy as np  # 导入 NumPy，用于数值计算与数组操作
import matplotlib.pyplot as plt  # 导入 pyplot 接口，用于绘制图形
from datetime import datetime  # 导入 datetime，用于生成时间戳
import os  # 导入 os，用于文件夹与路径操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 设置 Matplotlib 图形后端，适合桌面环境显示
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']  # 设置可用中文字体列表
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

n = 50  # 样本数量
X = np.linspace(0, 2 * np.pi, n)  # 在 [0, 2π] 区间内均匀生成输入样本
noise = np.random.normal(0, 0.2, n)  # 生成均值为 0、标准差为 0.2 的高斯噪声
y = np.sin(X) + noise  # 构造真实输出：正弦函数加噪声

print("真实规律: y = sin(x) + noise")  # 输出真实数据生成规律说明

# ============================
# 3. 初始化模型参数
# y_hat = w1*sin(x) + w2*cos(x) + b
# ============================
w1 = 0.0  # 初始化 sin(x) 项的权重参数
w2 = 0.0  # 初始化 cos(x) 项的权重参数
b  = 0.0  # 初始化偏置项

lr = 0.05  # 设置学习率（learning rate）
epochs = 20  # 设置训练轮数（epoch 数）

loss_history = []  # 用于记录每一轮训练的损失值
w1_history, w2_history, b_history = [], [], []  # 用于记录参数变化历史

# ============================
# 4. 全量梯度下降训练
# ============================
print("=" * 80)  # 打印分隔线
print("开始全量梯度下降训练（正弦非线性回归）")  # 输出训练开始提示
print("=" * 80)  # 打印分隔线

for epoch in range(epochs):  # 遍历训练轮数
    y_pred = w1 * np.sin(X) + w2 * np.cos(X) + b  # 根据当前参数计算模型预测值
    error = y_pred - y  # 计算预测值与真实值之间的误差
    loss = np.mean(error ** 2)  # 计算均方误差（MSE）作为损失函数

    dw1 = (2 / n) * np.sum(error * np.sin(X))  # 计算 w1 的梯度
    dw2 = (2 / n) * np.sum(error * np.cos(X))  # 计算 w2 的梯度
    db  = (2 / n) * np.sum(error)  # 计算 b 的梯度

    print(  # 打印当前轮训练信息
        f"第 {epoch+1:02d} 轮 | "
        f"Loss={loss:7.4f} | "
        f"w1={w1:6.3f}, w2={w2:6.3f}, b={b:6.3f}"
    )

    loss_history.append(loss)  # 保存当前轮损失
    w1_history.append(w1)  # 保存当前轮 w1
    w2_history.append(w2)  # 保存当前轮 w2
    b_history.append(b)  # 保存当前轮 b

    w1 -= lr * dw1  # 根据梯度下降规则更新 w1
    w2 -= lr * dw2  # 根据梯度下降规则更新 w2
    b  -= lr * db  # 根据梯度下降规则更新 b

print("=" * 80)  # 打印分隔线
print("训练结束")  # 输出训练结束提示
print("=" * 80)  # 打印分隔线

# ============================
# 5. 二维平面拟合可视化
# ============================
plt.figure(figsize=(7, 5))  # 创建绘图窗口并设置大小
plt.scatter(X, y, label="样本数据", alpha=0.7)  # 绘制样本散点图

# 中间过程曲线
for i in [0, 4, 9, 14, 19]:  # 选择若干关键训练轮次
    y_mid = (  # 计算第 i 轮对应的模型预测曲线
        w1_history[i] * np.sin(X) +
        w2_history[i] * np.cos(X) +
        b_history[i]
    )
    plt.plot(X, y_mid, "--", alpha=0.6, label=f"第 {i+1} 轮")  # 绘制中间过程曲线

# 最终模型
y_final = w1 * np.sin(X) + w2 * np.cos(X) + b  # 计算最终模型预测结果
plt.plot(X, y_final, color="black", linewidth=2, label="最终模型")  # 绘制最终拟合曲线

plt.xlabel("输入 x")  # 设置 x 轴标签
plt.ylabel("输出 y")  # 设置 y 轴标签
plt.title("全量梯度下降训练过程（正弦非线性回归）")  # 设置图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

fit_path = f"{out_dir}/sine_fit_{timestamp}.png"  # 构造拟合结果图像保存路径
plt.savefig(fit_path, dpi=600)  # 保存图像为高分辨率 PNG 文件
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口

# ============================
# 6. 损失下降曲线
# ============================
plt.figure(figsize=(6, 4))  # 创建损失曲线绘图窗口
plt.plot(loss_history, linewidth=2)  # 绘制损失随训练轮数变化曲线
plt.xlabel("训练轮数（Epoch）")  # 设置 x 轴标签
plt.ylabel("均方误差（MSE）")  # 设置 y 轴标签
plt.title("全量梯度下降的损失下降曲线（正弦回归）")  # 设置图像标题
plt.grid(True)  # 显示网格

loss_path = f"{out_dir}/sine_loss_{timestamp}.png"  # 构造损失曲线图像保存路径
plt.savefig(loss_path, dpi=600)  # 保存损失曲线图像
plt.show()  # 显示图像
plt.close()  # 关闭当前图像窗口

# ============================
# 7. 最终结果
# ============================
print("最终模型参数：")  # 输出提示信息
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")  # 输出最终模型参数
print("图像已保存：")  # 输出提示信息
print(fit_path)  # 输出拟合结果图像路径
print(loss_path)  # 输出损失曲线图像路径
