from game import Game2048
from agent import Agent

# board = [[256, 256, 64, 64], [2, 2, 4, 8], [4, 4, 2, 4], [0, 0, 0, 2]]
# game = Game2048(0.9)
# print(game.play_debug())

agent = Agent()
# ! warning number of episodes
agent.train(100)
