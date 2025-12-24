import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import seaborn as sns

from dqn import DQN
from game import Game2048
from replaybuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        # ! warning learning rate
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        deque_size=1000,
        batch_size=32,
        network_sync_rate=10,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.policy = DQN().to(self.device)
        self.target = DQN().to(self.device)

        self.target.load_state_dict(self.policy.state_dict())
        self.replay_buffer = ReplayBuffer(deque_size)
        self.board = Game2048()
        self.deque_size = deque_size
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.network_sync_rate = network_sync_rate

        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(), lr=self.learning_rate
        )
        self.rewards_per_episode: list[float] = []
        self.loss_fn = nn.MSELoss()

    def train(self, episodes) -> None:
        step_count = 0
        rewards = []
        max_tiles = []

        for i in range(episodes):
            print(f"{i = }, {self.epsilon = }")
            self.board.new_game()
            total_reward = 0
            iterations = 0
            valid_actions_count = 0

            while not self.board.is_game_over():
                curr_reward, is_valid_action = self.action()
                total_reward += curr_reward

                step_count += 1
                valid_actions_count += is_valid_action

                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.random_sample(self.batch_size)
                    self.optimize(batch)

                if step_count % self.network_sync_rate == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                iterations += 1

            print(
                "max tile: ", self.board.max_tile_value, "total reward: ", total_reward, "iterations: ", iterations, "invalid action count: ", iterations - valid_actions_count
            )
            rewards.append(total_reward)
            max_tiles.append(self.board.max_tile_value)

            self.epsilon = max(self.epsilon - 1 / episodes, 0)

        plt.subplot(121)
        sns.lineplot(rewards)
        plt.subplot(122)
        sns.lineplot(max_tiles)
        plt.show()

    def optimize(self, batch) -> None:
        curr_q_list = []
        target_q_list = []

        for init_state, next_state, action, reward, is_terminated in batch:
            init_state = torch.tensor(init_state, dtype=torch.float32).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

            if is_terminated:
                target = torch.tensor(reward).to(self.device)
            else:
                with torch.no_grad():
                    target = torch.tensor(
                        reward + self.discount_factor * self.target(next_state).max()
                    ).to(self.device)

            curr_q = self.policy(init_state)
            curr_q_list.append(curr_q)

            target_q = self.target(init_state)
            target_q[self.board.actions.index(action)] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(curr_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action(self) -> tuple[float, int]:
        if random.random() < self.epsilon:
            action = self.board.sample_actions()
        else:
            with torch.no_grad():
                action_idx = (
                    self.policy(
                        torch.tensor(self.board.get_state(), dtype=torch.float32).to(
                            self.device
                        )
                    )
                    .argmax()
                    .item()
                )
                action = self.board.actions[action_idx]

        init_state = self.board.get_state()
        reward, is_possible_action = self.board.play(action)

        if is_possible_action:
            self.board.place_new_block()
        is_terminated = self.board.is_game_over()
        next_state = self.board.get_state()

        self.replay_buffer.insert_transition(
            init_state, next_state, action, reward, is_terminated
        )

        return reward, int(is_possible_action)
