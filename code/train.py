from kaggle_environments import evaluate, make, utils
import numpy as np
from deepAgent import ScoreModel
from deepAgent import DeepAgent
from arena import Arena
from evolution import Evolution
from datetime import datetime
class Trainer:
  NUM_EPISODES = 1
  def __init__(self):
    # constants:
    self.config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    self.SURVIVAL_THRESHOLD = 0.2
    self.GAMMA = 0.01
    self.AGENTS_NUM = 50
    self.HOUSE_SIZE = 5
    self.LOOK_AHEAD = 1
    self.NUM_ITERATIONS = 200
    self.EVAL_NUM = self.NUM_ITERATIONS / 10
    self.SAVE_PATH = "../models/genetic_from_nil_"

    # variables:
    self.arena = Arena(self.AGENTS_NUM, self.HOUSE_SIZE)
    self.winnerArena = Arena(int(self.SURVIVAL_THRESHOLD * self.AGENTS_NUM), int(self.SURVIVAL_THRESHOLD * self.AGENTS_NUM))
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
    rewards0 = evaluate("connectx", [agent0, agent1], num_episodes=Trainer.NUM_EPISODES)
    rewards1 = evaluate("connectx", [agent1, agent0], num_episodes=Trainer.NUM_EPISODES)
    out0 = sum(r[0] for r in rewards0) / sum(r[0] + r[1] for r in rewards0)
    out1 = sum(r[1] for r in rewards1) / sum(r[0] + r[1] for r in rewards1)
    out = 0.5 * (out0 + out1)
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
    self.winnerArena.arrangeMatches()
    agents = [DeepAgent(net) for net in self.winners]
    matchOutcome = self.winnerArena.runTournement(agents, self.match)
    self.ranks = np.sum(matchOutcome, 1)
    randomScore =  [self.match(agent.play, "random")[0] for agent in agents]
    negamaxScore = [self.match(agent.play, "negamax")[0] for agent in agents]
    print("winner scores:")
    print("ranks:", self.ranks)
    print("random:", randomScore)
    print("negamax:", negamaxScore)        

  def train(self):
    print("\nTraining...")
    for i in range(self.NUM_ITERATIONS):
      print(datetime.now(), "- training iteration #%d:" % i)
      self.trainIteration()
      if i % self.EVAL_NUM == 0:
        print("   Iteration #%d:" % i)
        self.evaluate()
        winnerInd = np.argmax(self.ranks)
        self.winners[winnerInd].save(self.SAVE_PATH + str(i) + ".pt")

if __name__ == '__main__':
  trainer = Trainer()
  trainer.train()
