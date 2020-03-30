import numpy as np
from numpy import random as rand
from deepAgent import ScoreModel
from deepAgent import DeepAgent
from kaggle_environments import evaluate, make, utils

class Arena:
  def __init__(self, agentsNum, houseSize = 5):
    self.agentsNum = agentsNum
    self.matches = np.zeros(shape=(1, 2))
    self.houseSize = houseSize # number of player in each house in the tournement
    self.matchOutcome = np.zeros((self.agentsNum, self.agentsNum))

  def arrangeMatches(self):
    perm = rand.permutation(self.agentsNum)
    houses = np.reshape(perm, (-1, self.houseSize))
    for house in houses:
      for index in range(len(house)):
        tempMatches = np.concatenate((np.repeat(house[index], self.houseSize - index - 1), house[index + 1 :])).reshape((-1, 2), order='F')
        self.matches = np.concatenate((self.matches, tempMatches), 0)

    self.matches = (self.matches[1:]).astype(int)

  def runTournement(self, agents, matchFunction):
    for pair in self.matches:
      [out0, out1] = matchFunction(agents[pair[0]].play, agents[pair[1]].play)
      self.matchOutcome[pair[0], pair[1]] = out0
      self.matchOutcome[pair[1], pair[0]] = out1
    return self.matchOutcome

  @staticmethod
  def test():
    print(">> TESTING:")
    arena = Arena(10, 5)
    arena.arrangeMatches()
    print("\nmatch pairs:")
    print(arena.matches)
    config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    model = ScoreModel(config)
    agent = DeepAgent(model)
    arena.runTournement([agent, agent, agent, agent, agent, agent, agent, agent, agent, agent], match)
    print("\nmatches outcome:")
    print(arena.matchOutcome, "\n")
    return 0

def match(agent0, agent1):
  rewards = evaluate("connectx", [agent0, agent1], num_episodes=10)
  out = sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)
  return [out, 1 - out]

if __name__ == '__main__':
  Arena.test()
