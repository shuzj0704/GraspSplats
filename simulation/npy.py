import numpy as np
import matplotlib.pyplot as plt

data = np.load('/home/shu/3DGS/GraspSplats/simulation/camera_output/20250807_020754/images/0_depth.npy')

# 统计信息
print(f"Shape: {data.shape}, Dtype: {data.dtype}")
print(f"Valid range: {np.min(data[data > 0]):.2f}m ~ {np.max(data):.2f}m")  # 忽略0值

# 可视化有效深度区域
valid_depth = np.ma.masked_where(data <= 0, data)  # 屏蔽0值
plt.imshow(valid_depth, cmap='jet')
plt.colorbar(label='Depth (meters)')
plt.title('Valid Depth Map (Non-zero Values)')
plt.show()