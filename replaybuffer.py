from collections import deque
import random


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.deque: deque[tuple] = deque([], maxlen=max_size)

    def insert_transition(
        self, init_state, next_state, action, reward, is_terminated
    ) -> None:
        self.deque.append((init_state, next_state, action, reward, is_terminated))

    def random_sample(self, sample_size) -> list[tuple]:
        return random.sample(self.deque, sample_size)

    def __len__(self) -> int:
        return len(self.deque)
