import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# 数据：来自实验数据处理表格
# ============================================================
k = np.array([10, 15, 20, 25, 30, 35, 40, 45])          # 干涉级数
l2 = np.array([25.473, 37.344, 49.225, 60.949,           # 弦长平方
               72.880, 84.784, 96.826, 108.869])

# ============================================================
# 最小二乘法计算
# ============================================================
n = len(k)
k_mean = np.mean(k)
l2_mean = np.mean(l2)

S_xx = np.sum((k - k_mean) ** 2)
S_xy = np.sum((k - k_mean) * (l2 - l2_mean))

b = S_xy / S_xx
a = l2_mean - b * k_mean

l2_fit = a + b * k
residuals = l2 - l2_fit

# 相关系数 R²
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((l2 - l2_mean) ** 2)
r_squared = 1 - ss_res / ss_tot

# ============================================================
# 打印计算结果
# ============================================================
print("=" * 55)
print("最小二乘法计算结果")
print("=" * 55)
print(f"数据点数 n          = {n}")
print(f"k 平均值   (k̄)      = {k_mean}")
print(f"l² 平均值  (l̄²)     = {l2_mean:.5f}")
print(f"Σ(k-k̄)²   (Sxx)    = {S_xx:.4f}")
print(f"Σ(k-k̄)(l²-l̄²) (Sxy) = {S_xy:.4f}")
print(f"斜率 b              = {b:.6f}")
print(f"截距 a              = {a:.6f}")
print(f"相关系数 R²          = {r_squared:.6f}")
print(f"拟合方程: l² = {a:.4f} + {b:.4f}·k")
print("=" * 55)
print("\n中间计算量表：")
print(f"{'k':>6} {'l²':>10} {'k-k̄':>10} {'l²-l̄²':>12} {'(k-k̄)²':>12} {'(k-k̄)(l²-l̄²)':>16}")
for i in range(n):
    dk = k[i] - k_mean
    dl2 = l2[i] - l2_mean
    print(f"{k[i]:>6} {l2[i]:>10.3f} {dk:>10.4f} {dl2:>12.4f} {dk**2:>12.4f} {dk*dl2:>16.4f}")
print(f"{'Σ':>6} {'':>10} {'':>10} {'':>12} {S_xx:>12.4f} {S_xy:>16.4f}")

# ============================================================
# 可视化
# ============================================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 6))

k_line = np.linspace(5, 50, 100)
ax.scatter(k, l2, c='#2B579A', s=70, zorder=5, edgecolors='white', linewidth=1.2,
           label='实验数据点')
ax.plot(k_line, a + b * k_line, '-', color='#D32F2F', linewidth=2,
        label=f'拟合直线: $l^2 = {a:.3f} + {b:.3f}\\,k$\n$R^2 = {r_squared:.5f}$')
ax.set_xlabel('干涉级数 $k$', fontsize=13)
ax.set_ylabel('弦长平方 $l^2$ (mm$^2$)', fontsize=13)
ax.set_title('牛顿环实验 — 最小二乘法线性拟合', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlim(5, 50)

plt.tight_layout()
plt.savefig('least_squares_fit.png', dpi=200, bbox_inches='tight')
plt.show()
print("\n图表已保存为 least_squares_fit.png")
