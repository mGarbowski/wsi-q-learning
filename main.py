from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env


@dataclass
class RewardSystem:
    on_success: float
    on_hole: float
    on_nothing: float
    on_wall_hit: float


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
        self.epsilon_decay_rate = 1 / (n_episodes * epsilon_decay)
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
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def update_q_table(self, state: int, next_state: int, action: int, reward):
        next_q_value = self.q_table[next_state].max()
        delta = reward + self.discount_factor * next_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * delta

    def episode(self) -> float:
        total_rewards = 0
        state, _ = self.env.reset()

        episode_end = False
        while not episode_end:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.update_q_table(state, next_state, action, float(reward))

            episode_end = truncated or terminated
            state = next_state
            total_rewards += reward

        return total_rewards

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        """Train agent and return q_table and vector of rewards over episodes"""
        episode_rewards = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            episode_rewards[episode] = self.episode()
            self.decay_epsilon()

        return self.q_table, episode_rewards


def train_agent(size: int, slippery: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return q_table and vector of average rewards over episodes."""
    n_episodes = 1_000
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = 0.5
    final_epsilon = 0.03
    discount_factor = 0.9

    env = make_env(size=size, visible=False, slippery=slippery)
    agent = Agent(
        env=env,
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        start_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=final_epsilon
    )

    return agent.train()


def average_episode_rewards(size: int, n_runs: int = 25):
    average_rewards = np.zeros(1_000)
    success_rate = 0
    q_table = None
    for run in range(n_runs):
        q_table, rewards = train_agent(size, False)
        success_rate += sum(rewards) / len(rewards)
        average_rewards += rewards
    average_rewards /= n_runs
    success_rate /= n_runs
    print(f"Average success rate: {success_rate}")
    plot_rewards_over_episodes(average_rewards)
    present_optimal_policy(q_table, size, False)


def plot_rewards_over_episodes(rewards: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(rewards, 'r')
    plt.show()


def present_optimal_policy(q_table: np.ndarray, size: int, slippery: bool):
    env = make_env(size=size, slippery=slippery, visible=True)
    state, _ = env.reset()
    end = False
    while not end:
        action = q_table[state].argmax()
        _, _, terminated, truncated, _ = env.step(action)
        end = terminated or truncated


def make_env(size: int = 8, visible: bool = False, slippery: bool = False):
    map_name = f"{size}x{size}"
    render_mode = "human" if visible else None
    return gym.make('FrozenLake-v1', desc=None, map_name=map_name, is_slippery=slippery, render_mode=render_mode)


def main():
    # q_table, rewards = train_agent()
    # plot_rewards_over_episodes(rewards)
    # present_optimal_policy(q_table)

    average_episode_rewards(4, 25)


if __name__ == '__main__':
    main()
