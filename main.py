import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, Space


def choose_action(epsilon: float, action_space: Space, q_table: np.ndarray, state) -> int:
    if np.random.random() < epsilon:
        return action_space.sample()
    else:
        actions = q_table[state]
        max_q_value = actions.max()
        max_actions = [i for i in range(actions.size) if abs(actions[i] - max_q_value) < 0.01]
        return np.random.choice(max_actions)


def q_learning(
        env: Env,
        n_episodes: int,
        max_steps_per_episode: int,
        learning_rate: float,
        discount_rate: float,
        epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    rewards = np.zeros(n_episodes)
    epsilon_decay = 1 / (n_episodes)

    for episode in range(n_episodes):
        step = 0
        state, _ = env.reset()
        reached_end = False
        episode_reward = 0
        while step <= max_steps_per_episode and not reached_end:
            action = choose_action(epsilon, env.action_space, q_table, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            delta = reward + discount_rate * q_table[new_state].max() - q_table[state, action]
            q_table[state][action] += learning_rate * delta

            episode_reward += reward
            state = new_state
            reached_end = terminated or truncated
            step += 1

        epsilon = max(epsilon - epsilon_decay, 0.03)
        episode_reward /= step
        rewards[episode] = episode_reward

    return q_table, rewards


def present_optimal_policy(q_table: np.ndarray):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    end = False
    while not end:
        action = q_table[state].argmax()
        _, _, terminated, truncated, _ = env.step(action)
        end = terminated or truncated


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)

    q_table, rewards = q_learning(env, 10000, 500, 0.01, 0.9, 0.5)
    plt.plot(rewards)
    plt.show()

    present_optimal_policy(q_table)

    env.close()

if __name__ == '__main__':
    main()
