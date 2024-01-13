import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':
    main()
