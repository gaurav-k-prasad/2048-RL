import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import seaborn as sns

from ddqn import DDQN
from dqncnn import QNetworkCNN
from game import Game2048
from replaybuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        # ! warning learning rate
        learning_rate=1e-4,
        discount_factor=0.999,
        epsilon=1.0,
        deque_size=25000,
        batch_size=128,
        network_sync_rate=6000,
        epsilon_decay_rate=1 - 1e-4,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = QNetworkCNN().to(self.device)
        self.target = QNetworkCNN().to(self.device)

        self.target.load_state_dict(self.policy.state_dict())
        self.replay_buffer = ReplayBuffer(deque_size)
        self.board = Game2048()
        self.deque_size = deque_size
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.network_sync_rate = network_sync_rate
        self.epsilon_decay_rate = epsilon_decay_rate

        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(), lr=self.learning_rate
        )
        self.rewards_per_episode: list[float] = []
        self.loss_fn = nn.MSELoss()

    def train(self, episodes) -> None:
        step_count: int = 0
        rewards: list[float] = []
        max_tiles: list[float] = []

        for i in range(episodes):
            self.board.new_game()
            total_reward = 0
            iterations = 0
            valid_actions_count = 0
            policy_invalid_count = 0
            game_info = []
            least_epsilon = False

            while not self.board.is_game_over():
                curr_reward, is_valid_action, is_policy_invalid = self.action()
                game_info.append((curr_reward))
                total_reward += curr_reward
                policy_invalid_count += is_policy_invalid

                step_count += 1
                valid_actions_count += is_valid_action

                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.random_sample(self.batch_size)
                    self.optimize(batch)
                    self.epsilon = max(self.epsilon * self.epsilon_decay_rate, 0.05)
                    if self.epsilon <= 0.05:
                        least_epsilon = True
                        break

                if step_count % self.network_sync_rate == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                iterations += 1

            print(
                f"{i = }, {self.epsilon = }, max tile: {self.board.max_tile_value}, "
                f"total reward: {total_reward}, iterations: {iterations}, "
                f"invalid action count {iterations - valid_actions_count}, ",
                f"policy invalid count: {policy_invalid_count}",
            )
            rewards.append(total_reward)
            max_tiles.append(self.board.max_tile_value)
            print(self.board)

            if least_epsilon:
                break

        self.evaulate(100)
        # self.play_debug()
        # plt.subplot(121)
        # sns.lineplot(rewards)
        # plt.subplot(122)
        # sns.lineplot(max_tiles)
        # plt.show()

    def optimize(self, batch) -> None:
        init_states, next_states, actions, rewards, is_terminated_vals = list(
            zip(*batch)
        )
        init_states = torch.tensor(init_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(
            [self.board.actions.index(action) for action in actions]
        ).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        is_terminated_vals = torch.tensor(is_terminated_vals).to(self.device)

        not_terminated_next_states = []
        for i in range(len(batch)):
            if not is_terminated_vals[i]:
                not_terminated_next_states.append(next_states[i])

        not_terminated_next_states = torch.tensor(not_terminated_next_states).to(
            self.device
        )
        with torch.no_grad():
            next_q_policy = self.policy(not_terminated_next_states)
            not_terminated_q_actions = next_q_policy.argmax(dim=1)

            next_q_target = self.target(not_terminated_next_states)
            not_terminated_q_vals = next_q_target.gather(
                1, not_terminated_q_actions.unsqueeze(1)
            ).squeeze(1)

        targets = []

        not_terminated_idx = 0
        for i in range(len(batch)):
            if is_terminated_vals[i]:
                targets.append(rewards[i])
            else:
                targets.append(
                    rewards[i]
                    + self.discount_factor * not_terminated_q_vals[not_terminated_idx]
                )
                not_terminated_idx += 1

        targets = torch.tensor(targets).to(self.device)
        curr_q_vals = self.policy(init_states)
        target_q_vals = curr_q_vals.clone().detach()
        for i, action in enumerate(actions):
            target_q_vals[i, action] = targets[i]

        loss = self.loss_fn(curr_q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action(self) -> tuple[float, int, int]:
        is_random = False
        if random.random() < self.epsilon:
            action = self.board.sample_actions()
            is_random = True
        else:
            with torch.no_grad():
                q_vals = self.policy(
                    torch.tensor([self.board.get_state_cnn()], dtype=torch.float32).to(
                        self.device
                    )
                ).squeeze()

                action_idx = q_vals.argmax().item()
                action = self.board.actions[action_idx]

        init_state = self.board.get_state_cnn()
        reward, is_possible_action = self.board.play(action)
        is_policy_invalid = 0

        if is_possible_action:
            self.board.place_new_block()
        elif not is_possible_action and not is_random:
            is_policy_invalid = 1

        is_terminated = self.board.is_game_over()
        next_state = self.board.get_state_cnn()

        self.replay_buffer.insert_transition(
            init_state, next_state, action, reward, is_terminated
        )
        return reward, int(is_possible_action), is_policy_invalid

    def evaulate(self, games: int):
        result = []

        for i in range(games):
            patience = 3
            total_reward = 0
            self.board.new_game()
            while not self.board.is_game_over():
                action_idx: int = (
                    self.policy(
                        torch.tensor(
                            self.board.get_state_cnn(), dtype=torch.float32
                        ).to(self.device)
                    )
                    .argmax()
                    .item()
                )
                action = self.board.actions[action_idx]
                reward, possible = self.board.play(action)
                total_reward += reward

                if possible:
                    self.board.place_new_block()
                    patience = 3
                else:
                    patience -= 1

                if patience == 0:
                    result.append(-100)
                    break
            else:
                result.append(total_reward)

        sns.lineplot(result)
        plt.show()

    def play_debug(self) -> None:
        self.board.new_game()

        while not self.board.is_game_over():
            action_idx: int = (
                self.policy(
                    torch.tensor(self.board.get_state_cnn(), dtype=torch.float32).to(
                        self.device
                    )
                )
                .argmax()
                .item()
            )
            action = self.board.actions[action_idx]
            _, possible = self.board.play(action)
            print(action)
            print(self.board)
            if possible:
                self.board.place_new_block()
            else:
                print("INVALID")
            print("new block")
            print(self.board)

            input()
