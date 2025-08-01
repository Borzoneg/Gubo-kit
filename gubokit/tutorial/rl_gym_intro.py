import gymnasium as gym
import cv2
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Optional
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import logging

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def see_all_envs():
    gym.pprint_registry()

def cartpole():
    env = gym.make("CartPole-v1", render_mode="human")

    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
    print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

    # Box observation space (continuous values)
    print(f"Observation space: {env.observation_space}")  # Box with 4 values
    # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation

    
    input(f"Press enter to start observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()  # Random action for now - real agents will be smarter!

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")

    env.close()

def blackjack_tutorial():
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    # Create environment and agent
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()
    
    """test agent"""
    total_rewards = []

    test_episodes = 1000
    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    for _ in range(test_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {test_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

class TutorialGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, size: int=5, render_mode=None):
        self.size = size
        self.render_mode = render_mode
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }

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
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.size - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            print()

def register_custom_env():
    gym.register(
                        id="gymnasium_env/TutorialGridWorld-v0",
                        entry_point=TutorialGridWorldEnv,
                        max_episode_steps=300,  # Prevent infinite episodes
                    )

def tutorial_use_custom_grid_env():
    env = gym.make("gymnasium_env/TutorialGridWorld-v0", render_mode="human")
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    
    obs, info = env.reset(seed=42)  # Use seed for reproducible testing

    print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

    # Test each action type
    actions = [0, 1, 2, 3]  # right, up, left, down
    env.render()
    for action in actions:
        old_pos = obs['agent'].copy()
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = obs['agent']
        print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")
        env.render()

def recording_all_tutorial():
        # Configuration
    num_eval_episodes = 4
    env_name = "CartPole-v1"  # Replace with your environment

    # Create environment with recording capabilities
    env = gym.make(env_name, render_mode="rgb_array")  # rgb_array needed for video recording

    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="data/out/cartpole-agent",    # Folder to save videos
        name_prefix="eval",               # Prefix for video filenames
        episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    print(f"Videos will be saved to: cartpole-agent/")

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        episode_over = False
        while not episode_over:
            # Replace this with your trained agent's policy
            action = env.action_space.sample()  # Random policy for demonstration

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            episode_over = terminated or truncated

        print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

    env.close()

    # Print summary statistics
    print(f'\nEvaluation Summary:')
    print(f'Episode durations: {list(env.time_queue)}')
    print(f'Episode rewards: {list(env.return_queue)}')
    print(f'Episode lengths: {list(env.length_queue)}')

    # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Average episode length: {avg_length:.1f} steps')
    print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')

def recording_sparse_tutorial():
    # Training configuration
    training_period = 250           # Record video every 250 episodes
    num_training_episodes = 10_000  # Total training episodes
    env_name = "CartPole-v1"

    # Set up logging for episode statistics
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Create environment with periodic video recording
    env = gym.make(env_name, render_mode="rgb_array")

    # Record videos periodically (every 250 episodes)
    env = RecordVideo(
        env,
        video_folder="data/out/cartpole-training",
        name_prefix="training",
        episode_trigger=lambda x: x % training_period == 0  # Only record every 250th episode
    )

    # Track statistics for every episode (lightweight)
    env = RecordEpisodeStatistics(env)

    print(f"Starting training for {num_training_episodes} episodes")
    print(f"Videos will be recorded every {training_period} episodes")
    print(f"Videos saved to: cartpole-training/")

    for episode_num in range(num_training_episodes):
        obs, info = env.reset()
        episode_over = False

        while not episode_over:
            # Replace with your actual training agent
            action = env.action_space.sample()  # Random policy for demonstration
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # Log episode statistics (available in info after episode ends)
        if "episode" in info:
            episode_data = info["episode"]
            logging.info(f"Episode {episode_num}: "
                        f"reward={episode_data['r']:.1f}, "
                        f"length={episode_data['l']}, "
                        f"time={episode_data['t']:.2f}s")

            # Additional analysis for milestone episodes
            if episode_num % 1000 == 0:
                # Look at recent performance (last 100 episodes)
                recent_rewards = list(env.return_queue)[-100:]
                if recent_rewards:
                    avg_recent = sum(recent_rewards) / len(recent_rewards)
                    print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")

def ant_mojoco():
    env = gym.make('Ant-v5', render_mode="human")
    epochs = 50
    #Sets an initial state
    print(env.action_space)
    # Rendering our instance 300 times
    for epoc in range(epochs):
        env.reset()
        for t in range(300):
            action = env.action_space.sample()
            #renders the environment
            #Takes a random action from its action space 
            # aka the number of unique actions an agent can perform
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(observation)
    env.close()

if __name__ == "__main__":
    recording_sparse_tutorial()