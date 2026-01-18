import matplotlib  # 导入 matplotlib 主库
import numpy as np  # 导入 numpy，用于数值计算
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图接口
from datetime import datetime  # 导入 datetime，用于生成时间戳
import os  # 导入 os，用于文件夹与路径操作

# ============================
# Matplotlib 中文设置
# ============================
matplotlib.use("TkAgg")  # 指定图形后端，适合桌面环境
matplotlib.rcParams['font.family'] = 'sans-serif'  # 设置字体族
matplotlib.rcParams['font.sans-serif'] = [  # 设置中文字体
    'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =========================
# 1. 定义 Sigmoid 函数
# =========================
def sigmoid(x):  # Sigmoid 函数定义
    return 1 / (1 + np.exp(-x))  # Sigmoid 数学公式

# =========================
# 2. 定义 Swish 一阶导数
# =========================
def swish_derivative(x, beta):  # Swish 的一阶导数
    s = sigmoid(beta * x)  # 计算 σ(βx)
    return s + beta * x * s * (1 - s)  # Swish'(x) 的解析形式

# =========================
# 3. 构造输入变量 x
# =========================
x = np.linspace(-6, 6, 400)  # 在区间 [-6, 6] 内均匀采样
betas = [0.1, 1.0, 10.0]  # 不同 β 参数

# =========================
# 4. 创建 figures 文件夹（如不存在）
# =========================
save_dir = "figures"  # 图像保存目录
os.makedirs(save_dir, exist_ok=True)  # 若文件夹不存在则创建

# =========================
# 5. 时间戳命名
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
filename = f"swish_derivative_{timestamp}.png"  # 文件名
save_path = os.path.join(save_dir, filename)  # 完整保存路径

# =========================
# 6. 绘图
# =========================
plt.figure(figsize=(6, 4))  # 创建画布

for beta in betas:  # 逐个绘制不同 β 的曲线
    plt.plot(
        x,
        swish_derivative(x, beta),
        linewidth=2,
        label=rf'$\beta={beta}$'
    )

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # y=0 参考线
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # x=0 参考线

plt.xlim(-6, 6)  # x 轴范围
plt.ylim(-0.2, 1.2)  # y 轴范围
plt.xlabel('x')  # x 轴标签
plt.ylabel(r'$\frac{d}{dx}\,\mathrm{Swish}(x)$')  # y 轴标签
plt.title('Swish 一阶导数')  # 图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

# =========================
# 7. 保存并显示
# =========================
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 保存图像
plt.show()  # 显示图像

print(f"Figure saved to: {save_path}")  # 打印保存路径
