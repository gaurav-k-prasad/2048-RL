import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from ddqn import DDQN
from dqncnn import QNetworkCNN
from game import Game2048
from replaybuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        name_output,
        learning_rate=1e-3,
        discount_factor=0.999,
        epsilon=1.0,
        deque_size=500000,
        batch_size=128,
        network_sync_rate=5000,
        epsilon_decay_rate=0.99994,
        checkpoint_path=None,
        learning_mode="ann"
    ) -> None:        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DDQN().to(self.device) if learning_mode == "ann" else QNetworkCNN().to(self.device)
        self.target = DDQN().to(self.device) if learning_mode == "ann" else QNetworkCNN().to(self.device)

        self.replay_buffer = ReplayBuffer(deque_size)
        self.board = Game2048(0.9, learning_mode)
        self.deque_size = deque_size
        self.batch_size = batch_size
        self.name_output = name_output

        self.learning_mode = learning_mode
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.network_sync_rate = network_sync_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.start_episode = 0

        self.optimizer = torch.optim.Adam(
            params=self.policy.parameters(), lr=self.learning_rate
        )
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_episode = checkpoint.get("episode", 0) + 1 # Resume from next episode
            del checkpoint

        gc.collect()
        torch.cuda.empty_cache()
        self.target.load_state_dict(self.policy.state_dict())
        self.rewards_per_episode: list[float] = []
        self.loss_fn = nn.MSELoss()

    def train(self, episodes) -> None:
        step_count: int = 0
        avg_rewards: list[float] = []
        avg_max_tiles: list[float] = []
        avg_iterations: list[float] = []

        curr_rewards: list[float] = []
        curr_max_tiles: list[float] = []
        curr_iterations: list[int] = []

        for i in range(self.start_episode, episodes + 1):
            self.board.new_game()
            total_reward = 0
            iterations = 0

            valid_actions_count = 0
            policy_invalid_count = 0

            while not self.board.is_game_over():
                curr_reward, is_valid_action, is_policy_invalid = self.action()
                total_reward += curr_reward
                policy_invalid_count += is_policy_invalid

                step_count += 1
                valid_actions_count += is_valid_action

                if len(self.replay_buffer) >= 2000 and step_count % 3 == 0:
                    batch = self.replay_buffer.random_sample(self.batch_size)
                    self.optimize(batch)

                if step_count % self.network_sync_rate == 0:
                    self.target.load_state_dict(self.policy.state_dict())
                iterations += 1

            # print(
            #     f"{i = }, {self.epsilon = }, max tile: {self.board.max_tile_value}, "
            #     f"total reward: {total_reward}, iterations: {iterations}, "
            #     f"invalid action count {iterations - valid_actions_count}, ",
            #     f"policy invalid count: {policy_invalid_count}",
            # )
            self.epsilon = max(self.epsilon * self.epsilon_decay_rate, 0.05)
            curr_rewards.append(total_reward)
            curr_iterations.append(iterations)
            curr_max_tiles.append(self.board.max_tile_value)
            print(f"iteration: {i}")

            if i > 0 and i % 500 == 0:
                avg_reward = sum(curr_rewards) / len(curr_rewards)
                avg_iteration = sum(curr_iterations) / len(curr_iterations)
                avg_max_tile = sum(curr_max_tiles) / len(curr_max_tiles)

                print(
                    f"{i = }, {self.epsilon = }, max tile: {avg_max_tile}, "
                    f"average reward: {avg_reward}, avg iterations: {avg_iteration}, "
                    f"average max tile: {avg_max_tile}"
                )
                curr_rewards = []
                curr_iterations = []
                curr_max_tiles = []
                gc.collect()
                avg_rewards.append(avg_reward)
                avg_iterations.append(avg_iteration)
                avg_max_tiles.append(avg_max_tile)

            if i > 0 and i % 5000 == 0:
                print(self.board)
                checkpoint = {
                    "episode": i,
                    "model_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                }

                torch.save(checkpoint, f"checkpoint_{self.learning_mode}_{self.name_output}.pth")
                self.save_line_plot(f"reward_{self.name_output}", avg_rewards, i)
                self.save_line_plot(f"iterations_{self.name_output}", avg_iterations, i)
                self.save_line_plot(f"max tiles_{self.name_output}", avg_max_tiles, i)

        print(avg_rewards)
        print(avg_max_tiles)
        print(avg_iterations)

        # self.play_debug()
        # plt.subplot(121)
        # sns.lineplot(rewards)
        # plt.subplot(122)
        # sns.lineplot(max_tiles)
        # plt.show()

    def optimize(self, batch) -> None:
        states, next_states, actions, rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(
            1
        )

        pred_q_values = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + (self.discount_factor * next_q_values * (~dones))

        loss = self.loss_fn(pred_q_values, targets)
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
                    torch.tensor([self.board.get_state()], dtype=torch.float32).to(
                        self.device
                    )
                ).squeeze()

                action = q_vals.argmax().item()

        init_state = self.board.get_state()
        reward, is_possible_action = self.board.play(action)
        is_policy_invalid = 0

        if is_possible_action:
            self.board.place_new_block()
        elif not is_possible_action and not is_random:
            is_policy_invalid = 1

        is_terminated = self.board.is_game_over()
        next_state = self.board.get_state()

        self.replay_buffer.insert_transition(
            init_state,
            next_state,
            action,
            reward,
            is_terminated,
        )
        return reward, int(is_possible_action), is_policy_invalid

    def evaulate(self, games: int):
        result = []

        for i in range(games):
            patience = 3
            total_reward = 0
            self.board.new_game()
            while not self.board.is_game_over():
                action: int = (
                    self.policy(
                        torch.tensor(
                            self.board.get_state(), dtype=torch.float32
                        ).to(self.device)
                    )
                    .argmax()
                    .item()
                )
                reward, possible = self.board.play(action)
                print(self.board)
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

    def save_line_plot(self, name, values, episode_num):
        plt.figure(figsize=(10, 5))
        plt.title(f"Training Progress - Episode {episode_num}")
        
        plt.plot(values, alpha=0.3, color='blue', label=f'Raw {name}')
        
        plt.xlabel("Episode")
        plt.ylabel(f"Total {name}")
        plt.legend()
        
        # Save the file with the episode count in the name
        plt.savefig(f"{self.learning_mode}_images/{name}_plot_{episode_num}.png")
        plt.close()

    def play_debug(self) -> None:
        self.board.new_game()

        while not self.board.is_game_over():
            q_vals = self.policy(
                    torch.tensor([self.board.get_state()], dtype=torch.float32).to(
                        self.device
                    )
                ).squeeze()

            action = q_vals.argmax().item()
            _, possible = self.board.play(action)
            print(self.board.action_to_string(action))
            print(self.board)
            if possible:
                self.board.place_new_block()
            else:
                print("INVALID")
            print("new block")
            print(self.board)

            input()
