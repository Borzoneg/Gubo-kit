import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2

class MujocoCube(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", ], "render_fps": 30}
    def __init__(self, xspace: int=5.0, yspace: int=5.0, render_mode=None, mujoco_file: str=None):
        super().__init__()
        self.xspace = xspace
        self.yspace = yspace
        self.render_mode = render_mode
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1, 0 as "uninitialized" state
        self._agent_location = np.array([-1, -1, 0], dtype=np.float64)
        self._target_location = np.array([-1, -1, 0], dtype=np.float64)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.space_range_min, self.space_range_max = np.array([-self.xspace, -self.yspace, 0]), np.array([self.xspace, self.yspace, 1])
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float64),
                "target": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float64),
            }
        )

        self.action_space = gym.spaces.Discrete(9)
        step = 1e-2
        dstep = step / np.sqrt(2)
        self._action_to_direction = {
            0: np.array([ dstep,  dstep, 0]),  # Move right and up
            1: np.array([ step,       0, 0]),  # Move right
            2: np.array([ dstep, -dstep, 0]),  # Move right and down
            3: np.array([ 0,      -step, 0]),  # Move down
            4: np.array([-dstep, -dstep, 0]),  # Move left and down
            5: np.array([-step,       0, 0]),   # Move left
            6: np.array([-dstep,  dstep, 0]),   # Move left and up
            7: np.array([ 0,       step, 0]),   # Move up
            8: np.array([ 0,          0, step]),   # Jump TODO: implement a force
        }
        self.mujoco_model = mujoco.MjModel.from_xml_path(mujoco_file)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_agentj = self.mujoco_model.joint("agentjoint").id
        self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wallx1").id]] = [-(xspace+0.1), 0, 0] # 0.1=size of  agent
        self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wallx2").id]] = [ (xspace+0.1), 0, 0]
        self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wally1").id]] = [0, -(yspace+0.1), 0]
        self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wally2").id]] = [0,  (yspace+0.1), 0]
        self.mujoco_viewer = None # init in reset to have target in right position
        self.mujoco_renderer = None # init in reset to have target in right position

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed: int= None, options: dict=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._agent_location = np.hstack((self.np_random.uniform(self.space_range_min[:2], self.space_range_max[:2]), 0.1)) # no rnd for z

        self._target_location = self._agent_location # just to enter the loop
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.hstack((self.np_random.uniform(self.space_range_min[:2], self.space_range_max[:2]), 0)) # no rnd for z

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.mujoco_data.qpos[self.mujoco_agentj:self.mujoco_agentj+7] = np.hstack((self._agent_location, [1, 0, 0, 0]))
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("target").id]] = self._target_location
            # self.mujoco_viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
            self.mujoco_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.mujoco_cam)
            self.mujoco_cam.distance = 13
            self.mujoco_renderer = mujoco.Renderer(self.mujoco_model, width=1280, height=720)
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            self.mujoco_renderer.update_scene(self.mujoco_data, camera=self.mujoco_cam)
            rendered_img = self.mujoco_renderer.render()
            cv2.imshow("Render", rendered_img)
            cv2.waitKey(0)

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, self.space_range_min, self.space_range_max)
        
        terminated = np.linalg.norm(self._agent_location[:2] - self._target_location[:2]) < 1e-2

        truncated = False
        
        reward = 1 if terminated else 0 # TODO: code a proper one

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            self.mujoco_data.qpos[self.mujoco_agentj:self.mujoco_agentj+7] = np.hstack((self._agent_location, [1, 0, 0, 0]))
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # self.mujoco_viewer.sync()
            self.mujoco_renderer.update_scene(self.mujoco_data, camera=self.mujoco_cam)
            rendered_img = self.mujoco_renderer.render()
            cv2.imshow("Render", rendered_img)
            ans = chr(cv2.waitKey(1) & 0xff)
            if ans == "q":
                quit()

def train_mujoco_cube():
    gym.register(id="gymnasium_env/MujocoCube-v0", entry_point=MujocoCube, max_episode_steps=300)
    env = gym.make("gymnasium_env/MujocoCube-v0", render_mode="human", xspace=5.0, yspace=5.0, mujoco_file="files/mujoco_files/MujocoCubeRL.xml")
    print("Enter to start")
    obs, info = env.reset(seed=42)
    print(obs)
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # print(obs)

if __name__ == "__main__":
    train_mujoco_cube()