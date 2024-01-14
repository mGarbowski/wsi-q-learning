import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, Space


class Agent:
    def __init__(
            self,
            env: Env,
            n_episodes: int,
            learning_rate: float,
            discount_factor: float,
            start_epsilon: float,
            epsilon_decay: float,
            min_epsilon: float
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = start_epsilon

        state_size = env.observation_space.n
        action_size = env.action_space.n
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state: int):
        """epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.q_table[state].argmax()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def update_q_table(self, state: int, next_state: int, action: int, reward):
        next_q_value = self.q_table[next_state].max()
        delta = reward + self.discount_factor * next_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * delta


def train_agent() -> tuple[np.ndarray, np.ndarray]:
    """Return q_table and vector of average rewards over episodes."""
    n_episodes = 100_000
    max_steps_per_episode = 200
    learning_rate = 0.1
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes)
    final_epsilon = 0.1
    discount_factor = 0.9

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)
    agent = Agent(
        env=env,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        start_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=final_epsilon
    )

    episode_rewards = np.zeros(n_episodes)
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_end = False
        step = 0

        while step <= max_steps_per_episode and not episode_end:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update_q_table(state, next_state, action, float(reward))
            episode_end = truncated or terminated
            state = next_state
            step += 1
            episode_rewards[episode] += reward

        agent.decay_epsilon()

    return agent.q_table, episode_rewards


def average_episode_rewards(n_runs: int = 25):
    average_rewards = np.zeros(1_000)
    for run in range(n_runs):
        _, rewards = train_agent()
        average_rewards += rewards
    average_rewards /= n_runs
    plot_rewards_over_episodes(average_rewards)


def plot_rewards_over_episodes(rewards: np.ndarray):
    plt.plot(rewards)
    plt.show()


def present_optimal_policy(q_table: np.ndarray):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    end = False
    while not end:
        action = q_table[state].argmax()
        _, _, terminated, truncated, _ = env.step(action)
        end = terminated or truncated


def main():
    q_table, rewards = train_agent()
    plot_rewards_over_episodes(rewards)
    present_optimal_policy(q_table)


if __name__ == '__main__':
    main()
