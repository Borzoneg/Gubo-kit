import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import cv2
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class MujocoCubeEnv(gym.Env):
    metadata = {"render_modes": ["mujoco_render", "mujoco_viewer", "rgb_array"], "render_fps": 30}
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
        self.observation_space = gym.spaces.Box(low=np.concatenate([self.space_range_min, self.space_range_min]), # put the space range twice, one for the agent the other for the target
                                                high=np.concatenate([self.space_range_max, self.space_range_max]),
                                                dtype=np.float64)

        self.goal_thrshold = 1e-2
        # define actions and actionspace
        step = 5e-2
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
            # 8: np.array([ 0,          0, step]),   # Jump TODO: implement a force
        }
        self.action_space = gym.spaces.Discrete(len(self._action_to_direction.keys()))
        # if needed start render model and env
        if self.render_mode in ["mujoco_render", "mujoco_viewer", "rgb_array"]:
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
            if self.render_mode == "mujoco_viewer":
                self.mujoco_viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
            else:
                self.mujoco_renderer = mujoco.Renderer(self.mujoco_model, width=1280, height=720)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return np.concatenate([self._agent_location, self._target_location])

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

        if self.render_mode in ["mujoco_render", "mujoco_viewer", "rgb_array"]:
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
        terminated = (distance < self.goal_thrshold)
        norm_distance = distance / (np.linalg.norm([2 * self.xspace, 2 * self.yspace]))
        truncated = False
        if terminated:
            reward = 0.1
        else:
            reward = -(distance)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode in ["mujoco_render", "mujoco_viewer", "rgb_array"]:
            self.mujoco_data.qpos[self.mujoco_agentj:self.mujoco_agentj+7] = np.hstack((self._agent_location, [1, 0, 0, 0]))
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            if self.render_mode == "mujoco_viewer": # if human we want to show
                self.mujoco_viewer.sync()
            else:
                rendered_img = self.mujoco_renderer.render()
                self.mujoco_renderer.update_scene(self.mujoco_data, camera='top')
                if self.render_mode == "rgb_array": # if rgb array we need to return an image
                    return rendered_img
                    # return cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                elif self.render_mode == "mujoco_render": # if human we want to show
                    cv2.imshow("Render", rendered_img)
                    ans = chr(cv2.waitKey(1) & 0xff)
                    if ans == "q":
                        quit()
        else:
            print(f"Agent: {self._agent_location}, target: {self._target_location} info: {self._get_info()}")

class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, n_out),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

class PPO:
    def __init__(self, env, lr, gamma, clip, n_updates_per_iteration, n_in, n_out):
        self.env = env
        self.lr = lr                                                # Learning rate of actor optimizer
        self.gamma = gamma                                          # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = clip                                            # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.n_updates_per_iteration = n_updates_per_iteration      # times the model gets updated each time it gets updated

        self.n_in = n_in            # dim of features for in and out
        self.n_out = n_out          # dim of features for in and out

        self.model = ActorCritic(n_in, n_out)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.logp_buf = []
        self.val_buf = []

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs, value = self.model(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def store_transition(self, obs, action, reward, ep_over, log_prob, value):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.done_buf.append(ep_over)
        self.logp_buf.append(log_prob)
        self.val_buf.append(value)

    
    def compute_rtgs(self, rewards, dones):
        rtgs = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            rtgs.insert(0, discounted_sum)
        return torch.tensor(rtgs, dtype=torch.float32)

    def update(self):
        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32)
        acts = torch.tensor(self.act_buf, dtype=torch.int64)
        old_logps = torch.stack(self.logp_buf)
        values = torch.stack(self.val_buf).detach()
        rtgs = self.compute_rtgs(self.rew_buf, self.done_buf)

        advs = rtgs - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(self.n_updates_per_iteration):
            probs, value_preds = self.model(obs)
            dist = Categorical(probs)
            logps = dist.log_prob(acts)

            ratio = torch.exp(logps - old_logps.detach())
            clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs
            loss_actor = -torch.min(ratio * advs, clip_adv).mean()

            loss_critic = nn.functional.mse_loss(value_preds.squeeze(), rtgs)
            loss = loss_actor + 0.5 * loss_critic

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffers
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.logp_buf.clear()
        self.val_buf.clear()
    
def train_mujoco_cube(render_mode):
    """ ======= TRAINING HYPERPARAMS ======= """
    timesteps_per_batch = 2048
    n_updates_per_iteration = 5
    render_interval = 50
    t_training = 3e6
    max_episode_steps = 5000

    """ ======= ENV INIT ======= """
    gym.register(id="gymnasium_env/MujocoCube-v0", entry_point=MujocoCubeEnv, max_episode_steps=max_episode_steps)
    env = gym.make("gymnasium_env/MujocoCube-v0", render_mode=render_mode, xspace=5.0, yspace=5.0, mujoco_file="files/mujoco_files/MujocoCubeRL.xml")
    if render_mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder="data/out/mujoco_cube",
            name_prefix="training",
            episode_trigger=lambda x: x % render_interval == 0  # Only record every 250th episode
        )
    env = RecordEpisodeStatistics(env)
    
    """ ======= AGENT INIT ======= """
    agent = PPO(
        env=env,
        lr = 0.005,
        gamma = 0.95,
        clip = 0.2,
        n_updates_per_iteration=n_updates_per_iteration,
        n_in=env.observation_space.shape[0],
        n_out=env.action_space.n)

    t, crnt_ep = 0, 1
    distances = []
    while t < t_training:
        obs, info = env.reset()
        print(f"=====Episode {crnt_ep:02d}=====")
        print(f"starting with: {obs}")
        ep_over = False
        dist_ep = []
        while not ep_over:
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            dist_ep.append(info["distance"])
            if terminated:
                print("SUCCESS!")
            ep_over = terminated or truncated
            agent.store_transition(obs, action, reward, ep_over, log_prob, value)
            obs = next_obs
            # print(f"{t:06d}/{t_training:06d}", end='\r')
            t += 1
            if (t % timesteps_per_batch) == 0:
                agent.update()
            if "mujoco" in render_mode:
                env.render()
        crnt_ep += 1
        distances.append(dist_ep)
        print(f"finished with: {info}")
    env.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set render mode")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', action='store_const', const='mujoco_viewer', dest='render_mode', help='Render mode: human')
    group.add_argument('-r', action='store_const', const='rgb_array', dest='render_mode', help='Render mode: rgb_array')
    group.add_argument('-t', action='store_const', const='text', dest='render_mode', help='Render mode: text')

    parser.set_defaults(render_mode='mujoco_render')

    args = parser.parse_args()
    print(f"Render mode is: {args.render_mode}")
    model = train_mujoco_cube(args.render_mode)
    # test_trained_agent(model=model)