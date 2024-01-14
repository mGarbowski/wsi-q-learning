from dataclasses import dataclass

import numpy as np
from gymnasium import Env


@dataclass
class RewardSystem:
    on_success: float
    on_hole: float
    on_nothing: float
    on_wall_hit: float


@dataclass
class EnvironmentResponse:
    prev_state: int
    state: int
    reward: float
    terminated: bool
    truncated: bool


class Agent:
    def __init__(
            self,
            env: Env,
            reward_system: RewardSystem,
            n_episodes: int,
            learning_rate: float,
            discount_factor: float,
            start_epsilon: float,
            epsilon_decay: float,
            min_epsilon: float
    ):
        self.env = env
        self.reward_system = reward_system
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

    def calculate_reward(self, response: EnvironmentResponse):
        if response.terminated and response.reward == 1:
            return self.reward_system.on_success

        if response.terminated and response.reward == 0:
            return self.reward_system.on_hole

        if response.prev_state == response.state:
            return self.reward_system.on_wall_hit

        return self.reward_system.on_nothing

    def episode(self, training: bool = True) -> float:
        total_rewards = 0
        state, _ = self.env.reset()

        episode_end = False
        while not episode_end:
            action = self.choose_action(state)
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            response = EnvironmentResponse(state, next_state, float(env_reward), terminated, truncated)
            reward = self.calculate_reward(response)
            if training:
                self.update_q_table(state, next_state, action, reward)

            episode_end = truncated or terminated
            state = next_state
            total_rewards += env_reward

        return total_rewards

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        """Train agent and return q_table and vector of rewards over episodes"""
        episode_rewards = np.zeros(self.n_episodes)
        for episode in range(self.n_episodes):
            episode_rewards[episode] = self.episode()
            self.decay_epsilon()

        return self.q_table, episode_rewards

    def evaluate(self, n_episodes: int) -> float:
        successes = 0
        for _ in range(n_episodes):
            if self.episode(training=False) > 0:
                successes += 1

        return successes / n_episodes
