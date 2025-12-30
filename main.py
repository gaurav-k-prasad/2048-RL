from game import Game2048
from agent import Agent

agent = Agent("cnn_learning_rate_1e-3", checkpoint_path='checkpoint_cnn.pth', learning_mode="cnn")
agent.train(100_000)