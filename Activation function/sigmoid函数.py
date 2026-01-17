import matplotlib  # 导入 matplotlib 主库
import numpy as np  # 导入 numpy，用于数值计算
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图接口
from datetime import datetime  # 导入 datetime，用于生成时间戳
import os  # 导入 os，用于文件夹与路径操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 指定图形后端，适合桌面环境（如 PyCharm）
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置字体族为无衬线字体
matplotlib.rcParams['font.sans-serif'] = [  # 设置候选中文字体列表
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'  # 常见中文字体，防止中文乱码
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# =========================
# 1. 定义 Sigmoid 函数
# =========================
def sigmoid(x):  # 定义 Sigmoid 激活函数
    return 1 / (1 + np.exp(-x))  # Sigmoid 数学公式，将输入映射到 (0,1)

# =========================
# 2. 构造输入变量 x
# =========================
x = np.linspace(-8, 8, 400)  # 在区间 [-8, 8] 内均匀采样 400 个点作为输入
y = sigmoid(x)  # 计算 Sigmoid 激活函数的输出值

# =========================
# 3. 创建 figures 文件夹（如不存在）
# =========================
save_dir = "figures"  # 设置图像保存目录名称
os.makedirs(save_dir, exist_ok=True)  # 若文件夹不存在则创建，存在则忽略

# =========================
# 4. 时间戳命名
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化为时间戳
filename = f"sigmoid_{timestamp}.png"  # 使用时间戳生成唯一文件名
save_path = os.path.join(save_dir, filename)  # 拼接完整的图像保存路径

# =========================
# 5. 绘图
# =========================
plt.figure(figsize=(6, 4))  # 创建画布并设置尺寸
plt.plot(x, y, linewidth=2, label=r'$a=\frac{1}{1+e^{-x}}$')  # 绘制 Sigmoid 曲线并设置标签

plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # 绘制 x=0 的垂直参考虚线
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)  # 绘制 y=0.5 的水平参考虚线

plt.xlim(-8, 8)  # 设置 x 轴显示范围
plt.ylim(0, 1.05)  # 设置 y 轴显示范围
plt.xlabel('x')  # 设置 x 轴标签
plt.ylabel('a')  # 设置 y 轴标签（激活输出）
plt.title('Sigmoid 激活函数')  # 设置图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线

# =========================
# 6. 保存并显示
# =========================
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 以高分辨率保存图像到指定路径
plt.show()  # 显示绘制的图像窗口

print(f"Figure saved to: {save_path}")  # 在终端打印图像保存路径
