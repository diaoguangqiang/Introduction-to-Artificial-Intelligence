import matplotlib  # Matplotlib 主库，用于绘图后端控制
import numpy as np  # NumPy，用于数值计算
import matplotlib.pyplot as plt  # Matplotlib 的 pyplot 接口，用于画图
from datetime import datetime  # 用于生成时间戳
import os  # 用于文件和目录操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 指定图形后端，适合桌面环境（如 PyCharm）
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置字体族为无衬线
matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]  # 设置常见中文字体，防止中文乱码
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# ============================
# 1. 输出目录
# ============================
out_dir = "figures"  # 图像输出目录名称
os.makedirs(out_dir, exist_ok=True)  # 若目录不存在则创建，存在则不报错
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳，用于文件命名

# ============================
# 2. 构造二维样本数据
# ============================
np.random.seed(42)  # 固定随机种子，保证实验可复现

n = 50  # 样本数量
X = np.linspace(0, 10, n)  # 在 [0, 10] 区间内均匀生成 n 个输入样本
print(f"样本总数:{len(X)}, X形状:{X.shape}, 样本分布:{X}")
true_w, true_b = 2.5, 1.0  # 真实线性关系的斜率和截距（仅用于生成数据）
noise = np.random.normal(0, 1.0, n)  # 生成均值为 0、方差为 1 的高斯噪声
# noise = np.random.normal(0, 0.2, n) # 小噪声
# noise = np.random.normal(0, 5.0, n) # 大噪声
y = true_w * X + true_b + noise  # 真实输出 y = wx + b + 噪声

# ============================
# 3. 初始化模型参数
# ============================
w = 0.0  # 初始化模型参数：斜率 w
b = 0.0  # 初始化模型参数：截距 b
lr = 0.01  # 学习率（learning rate）
epochs = 20  # 训练轮数（epoch）

loss_history = []  # 用于保存每一轮的损失值
w_history = []  # 用于保存每一轮的 w
b_history = []  # 用于保存每一轮的 b

# ============================
# 4. 全量梯度下降训练（逐轮打印）
# ============================
print("=" * 80)
print("开始全量梯度下降训练（线性回归）")
print("=" * 80)

for epoch in range(epochs):  # 迭代 epochs 轮
    y_pred = w * X + b  # 根据当前参数计算模型预测值
    error = y_pred - y  # 预测值与真实值之间的误差
    loss = np.mean(error ** 2)  # 均方误差（MSE），衡量整体预测误差
    dw = (2 / n) * np.sum(error * X)  # 对 w 的梯度：所有样本共同决定
    db = (2 / n) * np.sum(error)  # 对 b 的梯度：所有样本共同决定

    # ---- 打印当前轮训练信息 ----
    print(
        f"第 {epoch + 1:02d} 轮 | "
        f"Loss = {loss:8.4f} | "
        f"w = {w:8.4f}, b = {b:8.4f} | "
        f"dw = {dw:8.4f}, db = {db:8.4f}"
    )  # 输出当前轮的损失、参数和梯度

    # ---- 保存历史 ----
    loss_history.append(loss)  # 记录当前轮损失
    w_history.append(w)  # 记录当前轮 w
    b_history.append(b)  # 记录当前轮 b

    # ---- 参数更新 ----
    w = w - lr * dw  # 按梯度下降方向更新 w
    b = b - lr * db  # 按梯度下降方向更新 b

print("=" * 80)
print("训练结束")
print("=" * 80)

# ============================
# 5. 样本分布 + 训练过程可视化
# ============================
plt.figure(figsize=(7, 5))  # 创建画布
plt.scatter(X, y, color="blue", label="样本数据")  # 绘制样本散点图

# 中间过程直线（不同训练轮次的模型）
for i in [0, 4, 9, 14, 19]:
    plt.plot(
        X,
        w_history[i] * X + b_history[i],  # 第 i 轮对应的模型直线
        linestyle="--",
        alpha=0.6,
        label=f"第 {i + 1} 轮模型"
    )

# 最终模型
plt.plot(X, w * X + b, color="black", linewidth=2, label="最终模型")

plt.xlabel("输入 x")  # x 轴标签
plt.ylabel("输出 y")  # y 轴标签
plt.title("全量梯度下降训练过程（线性回归）")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

fit_path = f"{out_dir}/training_process_{timestamp}.png"  # 图像保存路径
plt.savefig(fit_path, dpi=300)  # 保存高分辨率图片
plt.show()  # 显示图像
plt.close()  # 关闭当前图像

# ============================
# 6. 损失下降曲线
# ============================
plt.figure(figsize=(6, 4))
plt.plot(loss_history, linewidth=2)  # 绘制损失随训练轮数变化曲线
plt.xlabel("训练轮数（Epoch）")
plt.ylabel("均方误差（MSE）")
plt.title("全量梯度下降的损失下降曲线")
plt.grid(True)

loss_path = f"{out_dir}/loss_curve_{timestamp}.png"  # 损失曲线保存路径
plt.savefig(loss_path, dpi=300)
plt.show()
plt.close()
