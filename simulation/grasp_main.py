import time
import math
import os
import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np
from datetime import datetime

current_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_path, "scene.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 初始化渲染上下文
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(1200, 900, "mujoco", None, None)
glfw.make_context_current(window)

# 创建渲染上下文
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# 创建场景
scene = mujoco.MjvScene(model, maxgeom=1000)
depth_buffer = np.zeros((480, 640), dtype=np.float32)


def get_save_path(base_dir="/home/shu/3DGS/GraspSplats/simulation/camera_output"):
    """生成按 "年月日_时分秒" 组织的保存路径"""
    now = datetime.now()
    date_dir = now.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, date_dir, "images")
    os.makedirs(save_path, exist_ok=True)
    
    return save_path

save_path = get_save_path()
print(f"当前保存路径: {save_path}")


def init_camera(model, camera_name):
    """初始化相机"""
    camera = mujoco.MjvCamera()
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera.fixedcamid = cam_id
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    fovy = model.cam_fovy[cam_id]
    # print(f"Camera '{camera_name}' fovy: {fovy} degrees")
    return camera


def get_image(w, h, camera, image_type="rgb"):
    """图像获取函数"""
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(),
        None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    mujoco.mjr_render(viewport, scene, context)
    
    if image_type == "rgb":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    else:
        depth = np.zeros((h, w), dtype=np.float32)
        mujoco.mjr_readPixels(None, depth, viewport, context)
        return np.flipud(depth)


# initialize cameras
camera_names = [f"camera_{i:02d}" for i in range(1, 22)]  # 生成 camera_01 到 camera_21 的名称  22
cameras = {str(i): init_camera(model, name) for i, name in enumerate(camera_names)}  # 使用数字键
cameras["camera_end"] = init_camera(model, "camera_end")  # 添加 camera_end


# main simulation loop
def run_simulation():
    # 初始化状态
    if model.key_time is not None and model.key_qpos is not None:
        # 从XML的keyframe加载初始qpos
        initial_qpos = model.key_qpos[0, :model.nq].copy()
        initial_qvel = model.key_qvel[0, :model.nv].copy() if model.key_qvel is not None else np.zeros(model.nv)
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = np.zeros(model.nu) # 清零所有控制信号
        mujoco.mj_forward(model, data)
        print("Initial state locked:", data.qpos)

    image_saved = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            images = {}
            depth_maps = {}
            for name, camera in cameras.items():
                # 获取RGB图像
                images[name] = get_image(640, 480, camera)
                cv2.imshow(name, images[name])

                # 获取并处理深度图
                depth_norm = get_image(640, 480, camera, "depth")
                if np.max(depth_norm) <= 0:
                    print(f"Warning: Invalid depth data for {name}")
                    continue

                # 物理参数校正
                near, far = 0.1, 2.0
                actual_depth = near + (far - near) * depth_norm
                valid_depth = actual_depth[actual_depth < far]
                min_depth = np.min(valid_depth) if valid_depth.size > 0 else near
                max_depth = np.max(valid_depth) if valid_depth.size > 0 else far
                depth_normalized = np.clip((actual_depth - min_depth) / (max_depth - min_depth), 0, 1)
                depth_color = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # 存储数据
                depth_maps[name] = {
                    'actual': actual_depth,
                    'normalized': depth_normalized,
                    'color': depth_color
                }

                # 显示
                cv2.imshow(f"Depth_{name}", depth_color)

            # 统一保存（仅一次）
            if not image_saved:
                for name in cameras:
                    cv2.imwrite(f"{save_path}/{name}.png", images[name])
                    depth_data = depth_maps[name]
                    # 保存实际深度（毫米单位）
                    # cv2.imwrite(f"{save_path}/{name}_actual_depth.png", (depth_data['actual'] * 1000).astype(np.uint16))
                    # 保存归一化深度
                    cv2.imwrite(f"{save_path}/{name}_depth.png", (depth_data['normalized'] * 65535).astype(np.uint16))
                    # 保存伪彩色图
                    cv2.imwrite(f"{save_path}/{name}_depth_color.png", depth_data['color'])
                    # 保存原始数据
                    np.save(f"{save_path}/{name}_depth.npy", depth_data['actual'])
                    print(f"Saved images and depth data for {name} to {save_path}")
                image_saved = True

            if cv2.waitKey(1) == 27:
                break
            viewer.sync()


if __name__ == "__main__":
    run_simulation()
    cv2.destroyAllWindows()
    glfw.terminate()
