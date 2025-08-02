import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import tqdm
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class MujocoCubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, xspace: int=5.0, yspace: int=5.0, render_mode=None, mujoco_file: str=None):
        super().__init__()
        self.xspace = xspace
        self.yspace = yspace
        self.render_mode = render_mode
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1, 0 as "uninitialized" state
        self._agent_location = np.array([-1, -1, 0], dtype=np.float64)
        self._target_location = np.array([-1, -1, 0], dtype=np.float64)

        # define what the agent can observe
        self.space_range_min, self.space_range_max = np.array([-self.xspace, -self.yspace, 0]), np.array([self.xspace, self.yspace, 1])
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float64),
                "target": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float64),
            }
        )

        # define actions and actionspace
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
        # if needed start render model and env
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.mujoco_model = mujoco.MjModel.from_xml_path(mujoco_file)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            self.mujoco_agentj = self.mujoco_model.joint("agentjoint").id
            # placing walls
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wallx1").id]] = [-(xspace+0.1), 0, 0] # 0.1=size of  agent
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wallx2").id]] = [ (xspace+0.1), 0, 0]
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wally1").id]] = [0, -(yspace+0.1), 0]
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("wally2").id]] = [0,  (yspace+0.1), 0]
            # setting up camera, does not work when using viewer
            self.mujoco_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.mujoco_cam)
            self.mujoco_cam.distance = 13
            # init renderer or viewer (renderer we have more contorl whereas viewer is more gui styled)
            self.mujoco_renderer = mujoco.Renderer(self.mujoco_model, width=1280, height=720)
            # self.mujoco_viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)

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

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.mujoco_data.qpos[self.mujoco_agentj:self.mujoco_agentj+7] = np.hstack((self._agent_location, [1, 0, 0, 0]))
            self.mujoco_data.mocap_pos[self.mujoco_model.body_mocapid[self.mujoco_model.body("target").id]] = self._target_location

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
        distance = np.linalg.norm(self._agent_location[:2] - self._target_location[:2])
        terminated = (distance < 1e-2)

        truncated = False
        
        reward = -distance

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.mujoco_data.qpos[self.mujoco_agentj:self.mujoco_agentj+7] = np.hstack((self._agent_location, [1, 0, 0, 0]))
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # self.mujoco_viewer.sync()
            self.mujoco_renderer.update_scene(self.mujoco_data, camera=self.mujoco_cam)
            rendered_img = self.mujoco_renderer.render()
            if self.render_mode == "rgb_array": # if rgb array we need to return an image
                return rendered_img
                # return cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            else: # if human we want to show
                cv2.imshow("Render", rendered_img)
                ans = chr(cv2.waitKey(1) & 0xff)
                if ans == "q":
                    quit()
            
def train_mujoco_cube(render_mode):
    render_interval = 2
    num_training_episodes = 5
    gym.register(id="gymnasium_env/MujocoCube-v0", entry_point=MujocoCubeEnv, max_episode_steps=300)
    env = gym.make("gymnasium_env/MujocoCube-v0", render_mode=render_mode, xspace=5.0, yspace=5.0, mujoco_file="files/mujoco_files/MujocoCubeRL.xml")
    if render_mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder="data/out/mujoco_cube",
            name_prefix="training",
            episode_trigger=lambda x: x % render_interval == 0  # Only record every 250th episode
        )
    env = RecordEpisodeStatistics(env)
    for i in range(num_training_episodes):
        obs, info = env.reset()
        print(f"=====Episode {i:02d}=====")
        print(f"starting with: {obs}")
        ep_over = False
        while not ep_over:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(i)
            ep_over = terminated or truncated
            if render_mode == "human":
                env.render()
        print(f"finished with: {info}")
    env.close()

def train_mujoco_cube_ppo(render_mode):
    render_interval = 2
    num_episodes = 500
    max_steps = 300
    total_timesteps = num_episodes * max_steps
    gym.register(
        id="gymnasium_env/MujocoCube-v0",
        entry_point=MujocoCubeEnv,
        max_episode_steps=max_steps
    )

    # Create vectorized env for PPO
    def make_env():
        env = gym.make(
            "gymnasium_env/MujocoCube-v0",
            render_mode=render_mode,
            xspace=5.0,
            yspace=5.0,
            mujoco_file="files/mujoco_files/MujocoCubeRL.xml"
        )
        if render_mode == "rgb_array":
            env = RecordVideo(
                env,
                video_folder="data/out/mujoco_cube",
                name_prefix="training",
                episode_trigger=lambda ep_id: ep_id % render_interval == 0,
                disable_logger=True
            )
        env = RecordEpisodeStatistics(env)
        return env

    env = make_vec_env(make_env, n_envs=1)

    # Use PPO with MultiInputPolicy (since your obs is Dict)
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    env.close()
    return model

def test_trained_agent(model, render_mode="human"):
    env = gym.make(
        "gymnasium_env/MujocoCube-v0",
        render_mode=render_mode,
        xspace=5.0,
        yspace=5.0,
        mujoco_file="files/mujoco_files/MujocoCubeRL.xml"
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if render_mode == "human":
            env.render()
        elif render_mode == "rgb_array":
            frame = env.render()
            cv2.imshow("Agent", frame[..., ::-1])  # Convert RGB to BGR for OpenCV
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

    env.close()
    if render_mode == "rgb_array":
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set render mode")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-u', action='store_const', const='human', dest='render_mode', help='Render mode: human')
    group.add_argument('-r', action='store_const', const='rgb_array', dest='render_mode', help='Render mode: rgb_array')
    group.add_argument('-t', action='store_const', const='text', dest='render_mode', help='Render mode: text')

    parser.set_defaults(render_mode='human')

    args = parser.parse_args()
    print(f"Render mode is: {args.render_mode}")
    model = train_mujoco_cube_ppo(args.render_mode)
    test_trained_agent(model=model)