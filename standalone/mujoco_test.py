import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

model = MjModel.from_xml_path("/home/gu/mujoco-3.3.2/model/humanoid/humanoid.xml")
data = MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    input(">>>")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
