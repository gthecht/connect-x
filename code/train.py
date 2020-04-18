from kaggle_environments import evaluate, make, utils
import numpy as np
from deepAgent import ScoreModel
from deepAgent import DeepAgent
from arena import Arena
from evolution import Evolution

class Trainer:
  NUM_EPISODES = 5
  def __init__(self):
    # constants:
    self.config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    self.SURVIVAL_THRESHOLD = 0.1
    self.GAMMA = 0.1
    self.AGENTS_NUM = 100
    self.HOUSE_SIZE = 5
    self.LOOK_AHEAD = 0
    self.NUM_ITERATIONS = 50
    self.EVAL_NUM = self.NUM_ITERATIONS / 10

    # variables:
    self.arena = Arena(self.AGENTS_NUM, self.HOUSE_SIZE)
    self.evolution = Evolution(self.SURVIVAL_THRESHOLD, self.GAMMA)
    self.nets = []
    self.initNets()
    self.winners = []
    self.ranks = []

  def initNets(self):
    for i in range(self.AGENTS_NUM):
      self.nets.append(ScoreModel(self.config))
  
  @staticmethod
  def match(agent0, agent1):
    rewards = evaluate("connectx", [agent0, agent1], num_episodes=Trainer.NUM_EPISODES)
    out = sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)
    return [out, 1 - out]

  def trainIteration(self):
    self.arena.arrangeMatches()
    agents = [DeepAgent(net) for net in self.nets]
    matchOutcome = self.arena.runTournement(agents, self.match)
    ranks = np.sum(matchOutcome, 1)
    outcome = self.evolution.evolve(self.nets, ranks)
    self.nets = outcome["nets"]
    self.winners = outcome["winners"]
    self.ranks = outcome["ranks"]

  def evaluate(self):
    winnerArena = Arena(int(self.SURVIVAL_THRESHOLD * self.AGENTS_NUM))
    winnerArena.arrangeMatches()
    agents = [DeepAgent(net) for net in self.winners]
    matchOutcome = winnerArena.runTournement(agents, self.match)
    self.ranks = np.sum(matchOutcome, 1)
    randomScore =  [self.match(agent.play, "random")[0] for agent in agents]
    negamaxScore = [self.match(agent.play, "negamax")[0] for agent in agents]
    print("winner scores:")
    print("ranks | random | negamax:")
    print([self.ranks, randomScore, negamaxScore])

  def train(self):
    print("\nTraining...")
    for i in range(self.NUM_ITERATIONS):
      print("training iteration #%d:" % i)
      self.trainIteration()
      if i % self.EVAL_NUM == 0:
        print("\n\nIteration #%d:" % i)
        self.evaluate()

if __name__ == '__main__':
  trainer = Trainer()
  trainer.train()
