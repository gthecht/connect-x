import numpy as np
from numpy import random as rand
from kaggle_environments import evaluate, make, utils
from board import Board
import copy

class Agent:
  def __init__(self, model, trainMethod = 0):
    self.model = model
    self.trainMethod = trainMethod
  
  def play(self, observation, configuration, *args):
    self.board = Board(configuration)
    modelOutput = self.model(self.board, observation)
    return modelOutput

#%% Models:
def testModel(board, observation):
  return rand.randint(board.columns)

def simpleModel(board, observation, withScore = False):
  currentState = board.assessState(observation)
  scoreList = -np.inf * np.ones(board.columns)
  for playOption in np.transpose(np.where(currentState["playOptions"])):
    observationCopy = copy.deepcopy(observation)
    observationCopy["board"][playOption[0] * board.columns + playOption[1]] = observation["mark"]
    newState = board.assessState(observationCopy)
    scoreList[playOption[1]] = (newState["score"])
  
  bestPlay = int(np.argmax(scoreList)) # make this a random choice
  bestScore = np.max(scoreList)
  if (withScore): return { "play": bestPlay, "score": bestScore }
  return bestPlay

def doubleModel(board, observation, withScore = False):
  currentState = board.assessState(observation)
  scoreList = -np.inf * np.ones(board.columns)
  for playOption in np.transpose(np.where(currentState["playOptions"])):
    observationCopy = copy.deepcopy(observation)
    observationCopy["board"][playOption[0] * board.columns + playOption[1]] = observation["mark"]
    observationCopy["mark"] = 3 - observation["mark"]
    scoreList[playOption[1]] = - simpleModel(board, observationCopy, True)["score"]
  
  bestPlay = int(np.argmax(scoreList)) # make this a random choice
  bestScore = np.max(scoreList)
  if (withScore): return { "play": bestPlay, "score": bestScore }
  return bestPlay

def complexModel(board, observation, withScore = False, count = 1):
  DEEPLEVEL = 2
  currentState = board.assessState(observation)
  scoreList = -np.inf * np.ones(board.columns)
  for playOption in np.transpose(np.where(currentState["playOptions"])):
    observationCopy = copy.deepcopy(observation)
    observationCopy["board"][playOption[0] * board.columns + playOption[1]] = observation["mark"]
    if (count == DEEPLEVEL):
      newState = board.assessState(observationCopy)
      scoreList[playOption[1]] = (newState["score"])
    else:
      observationCopy["mark"] = 3 - observation["mark"]
      scoreList[playOption[1]] = - complexModel(board, observationCopy, True, count + 1)["score"]
  
  bestPlay = int(np.argmax(scoreList)) # make this a random choice
  bestScore = np.max(scoreList)
  if (withScore): return { "play": bestPlay, "score": bestScore }
  return bestPlay
#%% Test
def test():
  env = make("connectx", debug=True)
  env.render()
  env.reset()
  # Play as the first agent against default "random" agent.
  testAgent = Agent(complexModel)
  env.run([testAgent.play, "negamax"])
  env.render(mode="ipython", width=500, height=450)

test()

import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

my_agent = Agent(complexModel)
write_agent_to_file(my_agent.play, "submission.py")