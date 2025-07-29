import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

model = MjModel.from_xml_path("/home/gu/mujoco210/model/loop.xml")
data = MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
