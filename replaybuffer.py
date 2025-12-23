from collections import deque
import random


class ReplayBuffer:
    def __init__(self) -> None:
        self.deque: deque[tuple] = deque([])

    def insert_transition(self, transition) -> None:
        self.deque.append(transition)

    def random_sample(self, sample_size) -> list[tuple]:
        return random.sample(self.deque, sample_size)

    def __len__(self) -> int:
        return len(self.deque)
