import mujoco
import mujoco.viewer
import cv2

def import_file_and_view(filepath):
    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=1280, height=720)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    while True:
        mujoco.mj_step(model, data)
        print(cam.distance)
        cam.distance += .01
        renderer.update_scene(data, camera=cam)
        rendered_img = renderer.render()
        cv2.imshow("Render", rendered_img)
        ans = chr(cv2.waitKey(0) & 0xff)
        if ans == "q":
            break

if __name__ == "__main__":
    import_file_and_view('/home/gu/Gubo-kit/files/mujoco_files/MujocoCubeRL.xml')