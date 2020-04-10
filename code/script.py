import numpy as np
from numpy import random as rand
from scipy import signal as spSig
from scipy.ndimage import convolve
from kaggle_environments import evaluate, make, utils
import copy

class Board:
  def __init__(self, config):
    self.config = config
    self.timeout = config["timeout"]
    self.columns = config["columns"]
    self.rows = config["rows"]
    self.inarow = config["inarow"]
    self.steps = config["steps"]
    self.SCOREPOWER = 3

  def score(self, outcomeDict, openBelow):
    # return (max(np.amax(outcome) for name, outcome in outcomeDict.items()) / self.inarow)  ** self.SCOREPOWER
    # return sum(np.sum((outcome > 0) * np.power(outcome, self.SCOREPOWER)) for name, outcome in outcomeDict.items())
    return sum(np.sum((outcome > 0) * np.power(outcome, self.SCOREPOWER) / (openBelow[name] + 1)) for name, outcome in outcomeDict.items())

  def checkState(self, myBoard, opponentBoard, openBoard):
    gameFilters = { 
      "vertical": np.ones((self.inarow,1)),
      "horizontal": np.ones((1,self.inarow)),
      "mainDiag": np.eye(self.inarow),
      "subDiag": np.flipud(np.eye(self.inarow))
    }

    filtersOutCome = { name: spSig.convolve2d(myBoard - self.inarow * opponentBoard, filter, mode="valid") \
                      for name, filter in gameFilters.items() }
    
    openBelowFilters = {
      "vertical": np.concatenate((np.zeros((self.inarow, 1)), np.ones((self.rows - self.inarow, 1))), axis = 0),
      "horizontal": np.concatenate((np.zeros((1, self.inarow)), np.ones((self.rows - 1, self.inarow))), axis = 0),
      "mainDiag": np.tril(np.ones((self.rows, self.inarow)), -1),
      "subDiag": np.fliplr(np.tril(np.ones((self.rows, self.inarow)), -1))
    }

    openBelow = { name: np.flipud(convolve(1 * (openBoard), filter, mode="constant", cval=0, origin=[-int((self.rows ) / 2), \
                int((filter.shape[1] - 1) / 2)])) for name, filter in openBelowFilters.items() }
    openBelow = { name: matrix[0:filtersOutCome[name].shape[0], 0:filtersOutCome[name].shape[1]] for name, matrix in openBelow.items() }
    score = self.score(filtersOutCome, openBelow)
    return { "score": score,  "filtersOutCome": filtersOutCome }

  def playOptions(self, openBoard):
    playOptionsFilter = np.array([[-1], [1]])
    playOptions = spSig.convolve2d(openBoard, playOptionsFilter, mode='full', boundary='fill', fillvalue=0)
    playOptions = (playOptions[1:, :] == 1)
    return playOptions
    
  def assessState(self, observation):
    self.board = np.reshape(observation["board"], (self.rows, self.columns))
    myBoard = (self.board == observation["mark"])
    opponentBoard = (self.board == 3 - observation["mark"]) # so that if I'm 2 then they are 1 and the opposite
    openBoard = (self.board == 0)
    playOptions = self.playOptions(openBoard)
    myState = self.checkState(myBoard, opponentBoard, openBoard)
    opponentState = self.checkState(opponentBoard, myBoard, openBoard)
    score = myState["score"] - opponentState["score"]
    self.boardState = { 
      "score": score,
      "myBoard": myBoard,
      "opponentBoard": opponentBoard,
      "playOptions": playOptions,
      "myState": myState,
      "opponentState": opponentState
    }
    return self.boardState

#%% my agent    
def my_agent(observation, configuration):
  board = Board(configuration)
  return complexModel(board, observation)

def complexModel(board, observation, DEEPLEVEL = 4, withScore = False, count = 1):
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
      scoreList[playOption[1]] = - complexModel(board, observationCopy, DEEPLEVEL, True, count + 1)["score"]
  
  bestPlay = int(np.argmax(scoreList))
  bestScore = np.max(scoreList)
  if (withScore): return { "play": bestPlay, "score": bestScore }
  return bestPlay

#%% Evaluate agent:
def mean_reward(rewards):  return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

#%% Submission file
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")

#%% Validate submission
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")