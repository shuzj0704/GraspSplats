import numpy as np
from scipy.spatial.transform import Rotation


def calculate_and_print_pose(image_id, camera_id, name, pos, xyaxes):
    """
    根据输入的位置信息和方向向量计算四元数和平移向量，并打印结果。
    
    Args:
        image_id (str): 图像 ID
        camera_id (str): 相机 ID
        name (str): 名称
        pos (str): 平移向量字符串，格式为 "TX TY TZ"
        xyaxes (str): x 和 y 轴方向向量字符串，格式为 "x1 x2 x3 y1 y2 y3"
    """
    # 解析平移向量
    tx, ty, tz = map(float, pos.split())

    # 解析 x 和 y 轴方向向量
    xaxis, yaxis = np.fromstring(xyaxes, sep=" ").reshape(2, 3)
    xaxis /= np.linalg.norm(xaxis)  # 归一化
    yaxis /= np.linalg.norm(yaxis)  # 归一化

    # 计算 z 轴方向向量
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)  # 归一化

    # 构造旋转矩阵
    R = np.column_stack([xaxis, yaxis, zaxis])

    # 转换为四元数
    rot = Rotation.from_matrix(R)
    qx, qy, qz, qw = rot.as_quat()  # 注意顺序是 (x, y, z, w)

    # 打印结果
    # print(f"Quaternion: QW: {qw}, QX: {qx}, QY: {qy}, QZ: {qz}")
    # print(f"Translation: TX: {tx}, TY: {ty}, TZ: {tz}")
    print(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}")


def calculate_camera_params(camera_id, model="PINHOLE", width=1280, height=720, fov_x=90, fov_y=60):
    """
    计算相机的内参并输出结果。
    
    Args:
        camera_id (int): 相机 ID。
        model (str): 相机模型，默认为 "PINHOLE"。
        width (int): 图像宽度（像素）。
        height (int): 图像高度（像素）。
        fov_x (float): 水平视场角（度）。
        fov_y (float): 垂直视场角（度）。
    
    Returns:
        str: 格式化的相机参数字符串。
    """
    # 计算焦距
    fx = (width / 2) / np.tan(np.radians(fov_x) / 2)
    fy = (height / 2) / np.tan(np.radians(fov_y) / 2)

    # 计算主点
    cx = width / 2
    cy = height / 2

    # 格式化输出
    params = f"{camera_id} {model} {width} {height} {fx:.8f} {fy:.8f} {cx:.1f} {cy:.1f}"
    return params


# 示例调用
camera_id = "1"
image_id = "1"  # begin from 1
name = "0.png"
pos = "0.3 0.25 0.4"
xyaxes = "-0.640184 -0.768221 0.000000 0.512729 -0.427274 0.744678"

print("images.txt:")
calculate_and_print_pose(image_id, camera_id, name, pos, xyaxes)
calculate_and_print_pose("2", "1", "1.png", pos="0.3 -0.25 0.4", xyaxes="0.640184 -0.768221 0.000000 0.512729 0.427274 0.744678")
calculate_and_print_pose("3", "1", "2.png", pos="0.8 0.25 0.4", xyaxes="-0.780869 0.624695 0.000000 -0.460940 -0.576176 0.674949")
calculate_and_print_pose("4", "1", "3.png", pos="0.8 -0.25 0.4", xyaxes="0.780869 0.624695 -0.000000 -0.460940 0.576176 0.674949")

# 计算相机内参并输出
camera_id = 1
model = "PINHOLE"
width = 640
height = 480

print("camras.txt:")
camera_params = calculate_camera_params(camera_id, model, width=640, height=480, fov_x=45, fov_y=45)
print(camera_params)