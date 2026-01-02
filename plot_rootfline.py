import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# 方案 A: 使用支持数学符号的内置字体
plt.rcParams['font.family'] = 'DejaVu Sans' # 这是默认，如果不行尝试下一种
# 你的本机实测数据
PEAK_BW = 35.4       # GB/s (STREAM Triad)
PEAK_FLOPS = 1100.0  # GFLOPS (保守估计全核峰值)

# 你的 AVX2Add 程序坐标
# 假设 20M数据，时间 6700us -> ~3 GFLOPS
app_ai = 1/12        # 0.083 FLOP/Byte
app_ai2= 1600/96     # 16.67 FLOP/Byte 
app_ai3= 200/12       # 16.67 FLOP/Byte
app_ai4= 12800/768     # 16.67 FLOP/Byte
app_flops = 2.97      # GFLOPS
app_flops2 =19.59     # GFLOPS
app_flops3 = 56.67    # GFLOPS
app_flops4 = 101.8     # GFLOPS

# 构造坐标轴
ai_axis = np.logspace(-2, 2.5, 500)
roofline = np.minimum(PEAK_FLOPS, ai_axis * PEAK_BW)

plt.figure(figsize=(10, 6))
plt.loglog(ai_axis, roofline, 'r-', label='Roofline Boundary', lw=3)
plt.scatter([app_ai], [app_flops], color='blue', s=100, label='Your AVX2Add')
plt.scatter([app_ai2], [app_flops2], color='red', s=100, label='Your AVX2Add_100loops')
plt.scatter([app_ai3], [app_flops3], color='yellow', s=100, label='Your AVX2Add_100loops_unroll')
plt.scatter([app_ai4], [app_flops4], color='pink', s=100, label='Your AVX2Add_100loops_unroll_8ways')

# 标注脊点
ridge_point = PEAK_FLOPS / PEAK_BW
plt.axvline(x=ridge_point, color='green', linestyle='--', alpha=0.5)
plt.text(ridge_point*1.1, 1, f'Ridge Point: {ridge_point:.1f}', color='green')

plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
plt.ylabel('Performance (GFLOPS)')
plt.title('AMD Ryzen 9 7945HX Roofline Analysis')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.savefig('test.png')
plt.show()