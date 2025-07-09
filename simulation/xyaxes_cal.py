import numpy as np

def calculate_camera_frame(camera_pos, target_pos):
    """
    严格遵循MuJoCo相机坐标系规范：
    1. 相机视角沿局部坐标系-Z轴方向
    2. +X轴指向画面右侧
    3. +Y轴指向画面上方
    
    参数:
        camera_pos: 相机位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
        
    返回:
        xyaxes字符串 "x_x x_y x_z y_x y_y y_z"
    """
    # 转换为numpy数组
    cam = np.array(camera_pos)
    tgt = np.array(target_pos)
    
    # 1. 计算-Z轴方向（相机视角方向）
    z_axis = cam - tgt  # 注意这里是cam→tgt的相反方向
    z_axis /= np.linalg.norm(z_axis)
    
    # 2. 构建正交坐标系（MuJoCo规范要求）
    # 初始向上向量（尝试保持Y轴尽可能垂直）
    up_ref = np.array([0, 0, 1]) 
    
    # 计算X轴（画面右侧方向）
    x_axis = np.cross(up_ref, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:  # 处理视角垂直的情况
        x_axis = np.array([1, 0, 0])   # 默认右侧方向
    else:
        x_axis /= np.linalg.norm(x_axis)
    
    # 计算Y轴（画面上方方向）
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # 3. 转换为xyaxes格式（只需X和Y轴）
    xyaxes = np.concatenate([x_axis, y_axis])
    
    return " ".join([f"{val:.6f}" for val in xyaxes])

# 示例验证
camera_pos_1 = [0.3, 0.25, 0.4]
camera_pos_2 = [0.3, -0.25, 0.4]
camera_pos_3 = [0.8, 0.25, 0.4]
camera_pos_4 = [0.8, -0.25, 0.4]

target_pos = [0.6, 0, 0.05]
xyaxes_1 = calculate_camera_frame(camera_pos_1, target_pos)
xyaxes_2 = calculate_camera_frame(camera_pos_2, target_pos)
xyaxes_3 = calculate_camera_frame(camera_pos_3, target_pos)
xyaxes_4 = calculate_camera_frame(camera_pos_4, target_pos)

print(f'xyaxes="{xyaxes_1}"')
print(f'xyaxes="{xyaxes_2}"')
print(f'xyaxes="{xyaxes_3}"')
print(f'xyaxes="{xyaxes_4}"')