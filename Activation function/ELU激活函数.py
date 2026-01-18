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
# 1. 定义 ELU 函数
# =========================
def elu(x, alpha=1.0):  # 定义 ELU 激活函数，alpha 控制负区饱和值
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))  # 正区线性，负区指数平滑

# =========================
# 2. 构造输入变量 x
# =========================
x = np.linspace(-5, 5, 400)  # 在区间 [-5, 5] 内均匀采样
y = elu(x)  # 计算 ELU 激活函数输出

# =========================
# 3. 创建 figures 文件夹（如不存在）
# =========================
save_dir = "figures"  # 图像保存目录
os.makedirs(save_dir, exist_ok=True)  # 若文件夹不存在则创建

# =========================
# 4. 时间戳命名
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
filename = f"elu_{timestamp}.png"  # 文件名
save_path = os.path.join(save_dir, filename)  # 完整保存路径

# =========================
# 5. 绘图
# =========================
plt.figure(figsize=(6, 4))  # 创建画布
plt.plot(
    x, y,
    linewidth=2,
    label=r'$\mathrm{ELU}(x)$'
)  # 绘制 ELU 曲线

# 坐标轴居中（便于与 ReLU 对比）
plt.axhline(0, color='black', linewidth=1)  # x 轴
plt.axvline(0, color='black', linewidth=1)  # y 轴

plt.xlim(-5, 5)  # 设置 x 轴范围
plt.ylim(-2, 6)  # 设置 y 轴范围，展示负区饱和
plt.xlabel('x')  # x 轴标签
plt.ylabel('ELU(x)')  # y 轴标签
plt.title('ELU 激活函数')  # 图像标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

# =========================
# 6. 保存并显示
# =========================
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 保存图像
plt.show()  # 显示图像

print(f"Figure saved to: {save_path}")  # 打印图像保存路径
