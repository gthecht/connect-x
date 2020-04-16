import numpy as np
from numpy import random as rand
from kaggle_environments import evaluate, make, utils
from board import Board
import copy

#%% Model:
def model(board, observation, lookAhead = 0, withScore = False):
  currentState = board.assessState(observation)
  scoreList = -np.inf * np.ones(board.columns)
  for playOption in np.transpose(np.where(currentState["playOptions"])):
    observationCopy = copy.deepcopy(observation)
    observationCopy["board"][playOption[0] * board.columns + playOption[1]] = observation["mark"]
    if (lookAhead == 0):
      newState = board.assessState(observationCopy)
      scoreList[playOption[1]] = (newState["score"])
    else:
      observationCopy["mark"] = 3 - observation["mark"]
      scoreList[playOption[1]] = - model(board, observationCopy, lookAhead - 1, True)["score"]

  bestPlay = int(np.argmax(scoreList)) # make this a random choice
  bestScore = np.max(scoreList)
  if (withScore): return { "play": bestPlay, "score": bestScore }
  return bestPlay

#%% Agent class
class Agent:
  def __init__(self, lookAhead, scoreFunc = False):
    self.lookAhead = lookAhead
    self.scoreFunc = scoreFunc
  
  def play(self, observation, configuration, *args):
    self.board = Board(configuration, self.scoreFunc)
    modelOutput = model(self.board, observation, self.lookAhead)
    return modelOutput

  @staticmethod
  def test():
    env = make("connectx", debug=True)
    env.reset()
    testAgent = Agent(1)
    env.run([testAgent.play, "negamax"])
    env.render(mode="ipython", width=500, height=450)

if __name__ == '__main__':
  print("Agent test")
  Agent.test()
