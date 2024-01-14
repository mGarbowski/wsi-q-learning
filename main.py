from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from q_learning import RewardSystem, Agent

DEFAULT_REWARD_SYSTEM = RewardSystem(1, 0, 0, 0)
ALTERNATE_REWARD_SYSTEM_1 = RewardSystem(10, -5, 0, -1)
ALTERNATE_REWARD_SYSTEM_2 = RewardSystem(1, -10, 0, 0)


@dataclass
class Experiment:
    board_size: int
    slippery: bool
    reward_system: RewardSystem
    n_independent_runs: int = 25
    n_episodes: int = 1_000
    learning_rate: float = 0.01
    start_epsilon: float = 1.0
    epsilon_decay: float = 0.5
    min_epsilon: float = 0.03
    discount_factor: float = 0.9

    def train_agent(self):
        env = make_env(size=self.board_size, visible=False, slippery=self.slippery)
        agent = Agent(
            env=env,
            reward_system=self.reward_system,
            n_episodes=self.n_episodes,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            start_epsilon=self.start_epsilon,
            epsilon_decay=self.epsilon_decay,
            min_epsilon=self.min_epsilon
        )

        return agent.train()

    def averaged_rewards(self) -> np.ndarray:
        avg_rewards = np.zeros(self.n_episodes)
        for _ in range(self.n_independent_runs):
            _, rewards = self.train_agent()
            avg_rewards += rewards
        avg_rewards /= self.n_independent_runs
        return avg_rewards


@dataclass
class PlotData:
    data: np.ndarray
    color: str
    label: str


def compare_results(results: list[PlotData], plot_path: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for result in results:
        plt.plot(result.data, result.color, label=result.label)
    plt.legend()
    if plot_path is not None:
        plt.savefig(plot_path)
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


def experiment_1():
    ex_1 = Experiment(
        board_size=4,
        slippery=False,
        reward_system=DEFAULT_REWARD_SYSTEM,
    )

    ex_2 = Experiment(
        board_size=4,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
    )

    ex_3 = Experiment(
        board_size=4,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
    )

    r1 = ex_1.averaged_rewards()
    r2 = ex_2.averaged_rewards()
    r3 = ex_3.averaged_rewards()

    compare_results([
        PlotData(r1, "r", "default"),
        PlotData(r2, "b", "punish holes and walls"),
        PlotData(r3, "g", "harsh punishment for holes"),
    ], "./docs/plots/experiment_1.png")


def experiment_2():
    ex_1 = Experiment(
        board_size=8,
        slippery=False,
        reward_system=DEFAULT_REWARD_SYSTEM,
    )

    ex_2 = Experiment(
        board_size=8,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
    )

    ex_3 = Experiment(
        board_size=8,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
    )

    r1 = ex_1.averaged_rewards()
    r2 = ex_2.averaged_rewards()
    r3 = ex_3.averaged_rewards()

    compare_results([
        PlotData(r1, "r", "default"),
        PlotData(r2, "b", "punish holes and walls"),
        PlotData(r3, "g", "harsh punishment for holes"),
    ], "./docs/plots/experiment_2.png")


def experiment_3():
    ex_1 = Experiment(
        board_size=4,
        slippery=True,
        reward_system=DEFAULT_REWARD_SYSTEM,
        n_episodes=10_000
    )

    ex_2 = Experiment(
        board_size=4,
        slippery=True,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
        n_episodes=10_000
    )

    ex_3 = Experiment(
        board_size=4,
        slippery=True,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
        n_episodes=10_000
    )

    r1 = ex_1.averaged_rewards()
    r2 = ex_2.averaged_rewards()
    r3 = ex_3.averaged_rewards()

    compare_results([
        PlotData(r1, "r", "default"),
        PlotData(r2, "b", "punish holes and walls"),
        PlotData(r3, "g", "harsh punishment for holes"),
    ], "./docs/plots/experiment_3.png")


def main():
    experiment_1()
    # experiment_2()
    # experiment_3()


if __name__ == '__main__':
    main()
