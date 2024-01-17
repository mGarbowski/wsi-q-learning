from experiment_utils import Experiment, show_report
from q_learning import RewardSystem

DEFAULT_REWARD_SYSTEM = RewardSystem(1, 0, 0, 0, 0)
ALTERNATE_REWARD_SYSTEM_1 = RewardSystem(1, -5, -10, 0, -1)
ALTERNATE_REWARD_SYSTEM_2 = RewardSystem(1, -10, -10, 0, 0)


def compare_default_rewards_over_board_sizes():
    show_report("board_sizes", [
        Experiment(
            label="4x4",
            board_size=4,
            slippery=False,
            n_episodes=1000,
            reward_system=DEFAULT_REWARD_SYSTEM
        ),
        Experiment(
            label="8x8",
            board_size=8,
            slippery=False,
            n_episodes=1000,
            reward_system=DEFAULT_REWARD_SYSTEM
        ),
    ])


def comparing_reward_systems_not_slippery():
    show_report("reward_systems_no_slip", [
        Experiment(
            label="Default rewards",
            board_size=8,
            n_episodes=1000,
            slippery=False,
            reward_system=DEFAULT_REWARD_SYSTEM,
        ),
        Experiment(
            label="Reward system 1",
            board_size=8,
            n_episodes=1000,
            slippery=False,
            reward_system=ALTERNATE_REWARD_SYSTEM_1,
        ),
        Experiment(
            label="Reward system 2",
            board_size=8,
            n_episodes=1000,
            slippery=False,
            reward_system=ALTERNATE_REWARD_SYSTEM_2,
        )
    ])


def comparing_reward_systems_slippery():
    show_report("reward_systems_slip", [
        Experiment(
            label="Default rewards",
            slippery=True,
            n_episodes=10_000,
            reward_system=DEFAULT_REWARD_SYSTEM
        ),
        Experiment(
            label="Reward system 1",
            slippery=True,
            n_episodes=10_000,
            reward_system=ALTERNATE_REWARD_SYSTEM_1
        ),
        Experiment(
            label="Reward system 2",
            slippery=True,
            n_episodes=10_000,
            reward_system=ALTERNATE_REWARD_SYSTEM_2
        ),

    ])


def main():
    compare_default_rewards_over_board_sizes()
    comparing_reward_systems_not_slippery()
    comparing_reward_systems_slippery()


if __name__ == '__main__':
    main()
