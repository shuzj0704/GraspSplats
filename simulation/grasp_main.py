import time
import math
import os
import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np
from datetime import datetime

model = mujoco.MjModel.from_xml_path('scene.xml')
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


def get_save_path(base_dir="/home/shu/GraspSplats/simulation/camera_output"):
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


def save_image(image, camera_name, save_path):
    """ 图像保存函数 """
    filename = f"{save_path}/{camera_name}.png"
    cv2.imwrite(filename, image)
    return filename


cameras = {
    "camera_end": init_camera(model, "camera_end"),
    "camera_01": init_camera(model, "camera_01"),
    "camera_02": init_camera(model, "camera_02"),
    "camera_03": init_camera(model, "camera_03"),
    "camera_04": init_camera(model, "camera_04")
}


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
            # 保持初始状态
            data.qpos[:] = initial_qpos
            data.ctrl[:] = np.zeros(model.nu)
            mujoco.mj_step(model, data)
            
            # 获取并显示图像
            images = {}
            for name, camera in cameras.items():
                images[name] = get_image(640, 480, camera)
                cv2.imshow(name, images[name])
            # 获取并显示深度图
            depth = get_image(640, 480, cameras["camera_end"], "depth")
            if np.max(depth) > 0:
                cv2.imshow("Depth", depth / np.max(depth))
            # 保存图像（仅一次）
            if not image_saved:
                for name, img in images.items():
                    saved_path = save_image(img, name, save_path)
                    print(f"Saved: {saved_path}")
                image_saved = True
            
            if cv2.waitKey(1) == 27:  # ESC退出
                break
            viewer.sync()


if __name__ == "__main__":
    run_simulation()
    cv2.destroyAllWindows()
    glfw.terminate()
