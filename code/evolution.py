import numpy as np
import math
from deepAgent import ScoreModel

class Evolution:
  def __init__(self, SURVIVAL_THRESHOLD, GAMMA):
    self.winners = []
    self.genNumber = 0
    self.SURVIVAL_THRESHOLD = SURVIVAL_THRESHOLD
    self.CHILDREN_NUMBER = math.ceil(1 / self.SURVIVAL_THRESHOLD)
    self.GAMMA = GAMMA

  def evolve(self, nets, ranks):
    ranks = np.array(ranks)
    numSurvivors = int(self.SURVIVAL_THRESHOLD * len(nets))
    survivorIndices = np.argpartition(ranks, -numSurvivors)[-numSurvivors:]
    survivorIndices = survivorIndices[np.argsort(ranks[survivorIndices], -1)][::-1]
    ranks = ranks[survivorIndices]
    winners = [nets[i] for i in survivorIndices]
    nets = self.mutate(winners)
    return { "nets": nets, "winners": winners, "ranks": ranks }

  def mutate(self, studs):
    nets = studs[:]
    for net in studs:
      for i in range(self.CHILDREN_NUMBER - 1):
        nets.append(net.mutate(self.genNumber, self.GAMMA))
    self.genNumber += 1
    return nets

  @staticmethod
  def test():
    print("\nTest for evolution:")
    config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    nets = []
    for i in range(10):
      nets.append(ScoreModel(config))
    ranks = [1, 2, 3, 7, 3, 6, 8, 4, 2, 9]
    evolutionTest = Evolution(0.2, 0.1)
    evolveOutcome = evolutionTest.evolve(nets, ranks)
    print("evolution outcome:")
    print("ranks:")
    print(evolveOutcome["ranks"])
    return 0

if __name__ == '__main__':
  Evolution.test()
