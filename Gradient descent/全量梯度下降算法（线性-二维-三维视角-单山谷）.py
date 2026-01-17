import matplotlib # 导入 Matplotlib 主库，用于绘图后端与全局设置
import numpy as np # 导入 NumPy，用于数值计算
import matplotlib.pyplot as plt # 导入 pyplot，用于绘制图表
from datetime import datetime # 导入 datetime，用于生成时间戳
import os # 导入 os，用于目录与文件操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg") # 设置 Matplotlib 使用 TkAgg 后端，适合桌面环境
matplotlib.rcParams['font.family'] = 'sans-serif' # 设置字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'] # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示异常问题

# ============================
# 1. 输出目录与时间戳
# ============================
out_dir = "figures" # 设置图像输出目录
os.makedirs(out_dir, exist_ok=True) # 若目录不存在则创建
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 生成当前时间戳

# ============================
# 2. 构造模拟数据（线性回归）
# ============================
np.random.seed(42) # 固定随机种子，保证实验可复现

n = 100 # 样本数量
X = np.linspace(0, 10, n) # 在区间 [0, 10] 内生成 n 个输入样本
true_w, true_b = 2.5, 1.0 # 设置真实线性模型参数（仅用于生成数据）
y = true_w * X + true_b + np.random.normal(0, 1, n) # 生成带高斯噪声的观测数据

X_mat = np.c_[X, np.ones(n)] # 构造设计矩阵，将偏置项合并进矩阵

# ============================
# 3. 初始化参数
# ============================
w = np.array([0.0, 0.0]) # 初始化参数向量 [w, b]
lr = 0.01 # 学习率，控制每次更新步长
epochs = 100 # 训练轮数

loss_history = [] # 用于记录每一轮的损失值
param_history = [] # 用于记录每一轮的参数 [w, b]

# ============================
# 4. 全量梯度下降训练（MSE，严格凸）
# ============================
for epoch in range(epochs): # 进行 epochs 轮训练
    y_pred = X_mat @ w # 根据当前参数计算预测值
    error = y_pred - y # 计算预测值与真实值的误差
    loss = np.mean(error ** 2) # 计算均方误差（MSE）作为损失函数

    loss_history.append(loss) # 保存当前轮损失
    param_history.append(w.copy()) # 保存当前参数（拷贝，防止被覆盖）

    grad = (2 / n) * X_mat.T @ error # 计算损失函数对参数的梯度（全量样本）
    w = w - lr * grad # 沿负梯度方向更新参数

    if (epoch + 1) % 10 == 0: # 每 10 轮打印一次训练信息
        print(f"Epoch {epoch+1:03d} | 损失={loss:.4f} | 参数={w}") # 输出当前状态

param_history = np.array(param_history) # 将参数历史转换为 NumPy 数组

# ============================
# 5. 三维：线性回归的损失曲面（单一山谷）
# ============================
w_range = np.linspace(0, 4, 120) # 设置参数 w 的取值范围
b_range = np.linspace(-1, 3, 120) # 设置参数 b 的取值范围
W, B = np.meshgrid(w_range, b_range) # 构造参数空间网格

Loss_surface = np.zeros_like(W) # 初始化损失曲面数组

for i in range(W.shape[0]): # 遍历参数网格的行
    for j in range(W.shape[1]): # 遍历参数网格的列
        y_hat = W[i, j] * X + B[i, j] # 使用当前参数组合计算预测值
        Loss_surface[i, j] = np.mean((y_hat - y) ** 2) # 计算对应的 MSE 损失

# ============================
# 6. 三维可视化（损失曲面 + 梯度下降轨迹）
# ============================
fig = plt.figure(figsize=(9, 7)) # 创建三维绘图窗口
ax = fig.add_subplot(111, projection="3d") # 添加三维子图

#在三维坐标系中，以 (W, B) 作为“地面坐标”，以 Loss_surface 作为“高度”，把整个损失地形画成一张三维曲面。
ax.plot_surface(W, B, Loss_surface, cmap="viridis", alpha=0.75) # 绘制三维损失曲面
ax.plot(
    param_history[:, 0], # 梯度下降过程中 w 的变化轨迹
    param_history[:, 1], # 梯度下降过程中 b 的变化轨迹
    loss_history, # 对应的损失值
    color="red", # 设置轨迹颜色
    marker="o", # 使用圆点标记轨迹
    markersize=3, # 设置标记大小
    linewidth=2, # 设置轨迹线宽
    label="梯度下降轨迹" # 设置图例说明
)

ax.set_xlabel("参数 w") # 设置 x 轴标签
ax.set_ylabel("参数 b") # 设置 y 轴标签
ax.set_zlabel("损失值（MSE）") # 设置 z 轴标签
ax.set_title("线性回归的三维损失曲面与梯度下降轨迹（单一山谷）") # 设置图像标题
ax.legend() # 显示图例

surface_path = f"{out_dir}/loss_surface_convex_{timestamp}.png" # 设置三维图像保存路径
plt.savefig(surface_path, dpi=600) # 保存三维图像
plt.show() # 显示图像
plt.close() # 关闭图像窗口

# ============================
# 7. 最终拟合结果（用于对照）
# ============================
plt.figure(figsize=(6, 4)) # 创建画布
plt.scatter(X, y, alpha=0.6, label="样本数据") # 绘制样本散点图
plt.plot(X, w[0] * X + w[1], color="black", linewidth=2, label="GD 拟合结果") # 绘制最终拟合直线
plt.xlabel("输入 X") # 设置 x 轴标签
plt.ylabel("输出 y") # 设置 y 轴标签
plt.title("线性回归的最终拟合结果（全量梯度下降）") # 设置标题
plt.legend() # 显示图例
plt.grid(True) # 显示网格

fit_path = f"{out_dir}/fitted_model_{timestamp}.png" # 设置拟合结果图像保存路径
plt.savefig(fit_path, dpi=600) # 保存图像
plt.close() # 关闭图像窗口

print("\n图像已保存：") # 输出提示信息
print(surface_path) # 输出三维损失曲面图路径
print(fit_path) # 输出最终拟合结果图路径
