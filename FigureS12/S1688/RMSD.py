import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# 设置非交互式后端（仅保存图像，不显示）
matplotlib.use('Agg')

# ======================
# 1. 全局样式设置
# ======================
# 1. 全局字体设置（多备选方案）
font_options = [
    'Times New Roman',
    'Times',
    'Liberation Serif',
    'DejaVu Serif',
    'serif'
]
rcParams['font.family'] = font_options
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.weight'] = 'bold'

# ======================
# 2. 数据准备
# ======================
try:
    data = np.loadtxt('rmsd-716.dat', comments=['#', '%'])
    time_ns = data[:, 0]/50  # 帧数转ns
    rmsd = data[:, 1]         # RMSD值
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit(1)

# ======================
# 3. 创建图形
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('white')  # 设置图形背景色

# ======================
# 4. 绘制曲线
# ======================
ax.plot(time_ns, rmsd, 
        color='#2ca02c',    # 科学绿
        linewidth=3.0,      # 线宽
        solid_capstyle='round',  # 线端圆角
        label='S1688')

# ======================
# 5. 坐标轴设置
# ======================
ax.set_xlim(-5, 205)
ax.set_ylim(0, 8)
ax.set_xlabel("Time (ns)", fontsize=18, fontweight='bold', labelpad=8)
ax.set_ylabel("RMSD (Å)", fontsize=18, fontweight='bold', labelpad=8)

# ======================
# 6. 刻度设置
# ======================
ax.tick_params(
    axis='both',
    which='major',
    labelsize=18,
    width=1.5,       # 刻度线宽度
    length=5         # 刻度线长度
)
# 单独设置刻度标签加粗
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# ======================
# 7. 图例设置
# ======================
# 图例设置（无边框+内部右上角）
legend = ax.legend(
    fontsize=14,
    frameon=False,  # 关闭边框
    loc='upper right',  # 图形内部右上角
    bbox_to_anchor=(0.98, 0.98),  # 微调位置（靠近右上角）
    handletextpad=0.5,  # 图例句柄与文本间距
    borderaxespad=0.5   # 图例与边框间距
)
# 手动设置图例字体加粗
for text in legend.get_texts():
    text.set_fontweight('bold')

# ======================
# 8. 边框设置
# ======================
for spine in ax.spines.values():
    spine.set_linewidth(3.0)  # 统一设置所有边框为3.0磅

# ======================
# 9. 网格设置
# ======================
ax.grid(
    True,
    linestyle=':',      # 虚线
    linewidth=0.8,      # 线宽
    alpha=0.4,          # 透明度
    color='gray'        # 网格颜色
)

# ======================
# 10. 布局调整
# ======================
plt.subplots_adjust(
    left=0.1,      # 左边距
    right=0.85,    # 为图例留出空间
    bottom=0.12,
    top=0.95
)

# ======================
# 11. 输出
# ======================
output_file = 'rmsd_professiona.png'
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
print(f"图形已保存到: {output_file}")

# 移除plt.show()，避免在非交互式环境中报错
# plt.show()