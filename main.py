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
    label: str
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
    n_evaluate_episodes: int = 100

    def make_agent(self) -> Agent:
        env = make_env(size=self.board_size, visible=False, slippery=self.slippery)
        return Agent(
            env=env,
            reward_system=self.reward_system,
            n_episodes=self.n_episodes,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            start_epsilon=self.start_epsilon,
            epsilon_decay=self.epsilon_decay,
            min_epsilon=self.min_epsilon
        )

    def averaged_results(self) -> tuple[np.ndarray, float]:
        avg_rewards = np.zeros(self.n_episodes)
        avg_success_rate = 0
        for _ in range(self.n_independent_runs):
            agent = self.make_agent()
            _, rewards = agent.train()
            avg_rewards += rewards
            avg_success_rate += agent.evaluate(self.n_evaluate_episodes)
        avg_rewards /= self.n_independent_runs
        avg_success_rate /= self.n_independent_runs
        return avg_rewards, avg_success_rate


@dataclass
class PlotData:
    data: np.ndarray
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
        plt.plot(result.data, label=result.label)
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
        label="Default rewards",
        board_size=4,
        slippery=False,
        reward_system=DEFAULT_REWARD_SYSTEM,
    )

    ex_2 = Experiment(
        label="Reward system 1",
        board_size=4,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
    )

    ex_3 = Experiment(
        label="Reward system 2",
        board_size=4,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
    )

    experiments = [ex_1, ex_2, ex_3]
    results = [ex.averaged_results() for ex in experiments]

    print("Success rates:")
    for ex, (_, sr) in zip(experiments, results):
        print(f"{ex.label}: {sr * 100:.2f}%")

    compare_results([
        PlotData(r, ex.label)
        for ex, (r, _) in zip(experiments, results)
    ], "./docs/plots/experiment_1.png")


def experiment_2():
    ex_1 = Experiment(
        label="Default rewards",
        board_size=8,
        slippery=False,
        reward_system=DEFAULT_REWARD_SYSTEM,
    )

    ex_2 = Experiment(
        label="Reward system 1",
        board_size=8,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
    )

    ex_3 = Experiment(
        label="Reward system 2",
        board_size=8,
        slippery=False,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
    )

    experiments = [ex_1, ex_2, ex_3]
    results = [ex.averaged_results() for ex in experiments]

    print("Success rates:")
    for ex, (_, sr) in zip(experiments, results):
        print(f"{ex.label}: {sr * 100:.2f}%")

    compare_results([
        PlotData(r, ex.label)
        for ex, (r, _) in zip(experiments, results)
    ], "./docs/plots/experiment_2.png")


def experiment_3():
    ex_1 = Experiment(
        label="Default rewards",
        board_size=4,
        slippery=True,
        reward_system=DEFAULT_REWARD_SYSTEM,
        n_episodes=10_000
    )

    ex_2 = Experiment(
        label="Reward system 1",
        board_size=4,
        slippery=True,
        reward_system=ALTERNATE_REWARD_SYSTEM_1,
        n_episodes=10_000
    )

    ex_3 = Experiment(
        label="Reward system 2",
        board_size=4,
        slippery=True,
        reward_system=ALTERNATE_REWARD_SYSTEM_2,
        n_episodes=10_000
    )

    experiments = [ex_1, ex_2, ex_3]
    results = [ex.averaged_results() for ex in experiments]

    print("Success rates:")
    for ex, (_, sr) in zip(experiments, results):
        print(f"{ex.label}: {sr * 100:.2f}%")

    compare_results([
        PlotData(r, ex.label)
        for ex, (r, _) in zip(experiments, results)
    ], "./docs/plots/experiment_3.png")


def main():
    experiment_1()
    experiment_2()
    experiment_3()


if __name__ == '__main__':
    main()
