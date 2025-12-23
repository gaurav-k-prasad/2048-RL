from dqn import DQN
from game import Game2048
from replaybuffer import ReplayBuffer


class Agent:
  def __init__(self) -> None:
    self.policy = DQN()
    self.target = DQN()

    self.target.load_state_dict(self.policy.state_dict())
    self.replay_buffer = ReplayBuffer()
    self.board = Game2048()
    self.deque_size = 512
    self.batch_size = 32

  def train(self, epoch):
    ...
  
  def data_collection(self):
    self.board.new_game()


