import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class MujocoCube(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", ], "render_fps": 30}
    def __init__(self, xspace: int=5.0, yspace: int=5.0, render_mode=None, mujoco_file: str=None):
        super().__init__()
        self.xspace = xspace
        self.yspace = yspace
        self.render_mode = render_mode
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1, 0 as "uninitialized" state
        self._agent_location = np.array([-1, -1, 0], dtype=np.float32)
        self._target_location = np.array([-1, -1, 0], dtype=np.float32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.space_range_min, self.space_range_max = np.array([-self.xspace, -self.yspace, 0]), np.array([self.xspace, self.yspace, 1])
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float32),
                "target": gym.spaces.Box(low=self.space_range_min, high=self.space_range_max, dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Discrete(5)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([ .1,  0, 0]),  # Move right
            1: np.array([ 0,  .1, 0]),  # Move up
            2: np.array([-.1,  0, 0]),  # Move left
            3: np.array([ 0, -.1, 0]),  # Move down
            4: np.array([ 0,  0, .1]),  # Jump TODO: implement a force
        }
        self.mujoco_model = mujoco.MjModel.from_xml_path(mujoco_file)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.mujoco_agent_id = self.mujoco_model.body("agent").id
        self.mujoco_target_mocap_id = self.mujoco_model.body_mocapid[self.mujoco_model.body("target").id]
        self.mujoco_data.body("wallx1").xpos = [-xspace, 0, 0]
        self.mujoco_data.body("wallx2").xpos = [ xspace, 0, 0]
        self.mujoco_data.body("wally1").xpos = [0, -yspace, 0]
        self.mujoco_data.body("wally2").xpos = [0,  yspace, 0]
        print(self.mujoco_data.body("wallx1").xpos)
        print(self.mujoco_data.body("wallx2").xpos)
        print(self.mujoco_data.body("wally1").xpos)
        print(self.mujoco_data.body("wally2").xpos)
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        self.mujoco_renderer = mujoco.Renderer(self.mujoco_model)
        self.mujoco_viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)

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

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.uniform(self.space_range_min, self.space_range_max)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location # just to enter the loop
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(self.space_range_min, self.space_range_max)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # self.mujoco_data.mocap_pos[self.mujoco_target_mocap_id] = self._target_location
            self.mujoco_data.body("target").xpos = self._target_location
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-4) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(self._agent_location + direction, self.space_range_min, self.space_range_max)

        # Check if agent reached the target
        terminated = np.linalg.norm(self._agent_location - self._target_location) < 1e-3

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0 # TODO: code a proper one

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            self.mujoco_data.body("agent").xpos = self._agent_location
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
            self.mujoco_renderer.render()

def train_mujoco_cube():
    gym.register(id="gymnasium_env/MujocoCube-v0", entry_point=MujocoCube, max_episode_steps=300)
    env = gym.make("gymnasium_env/MujocoCube-v0", render_mode="human", xspace=500.0, yspace=500.0, mujoco_file="files/mujoco_files/MujocoCubeRL.xml")
    obs, info = env.reset(seed=42)
    print(obs)

    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(obs)

if __name__ == "__main__":
    train_mujoco_cube()