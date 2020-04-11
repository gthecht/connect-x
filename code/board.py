import numpy as np
from scipy import signal as spSig
from scipy.ndimage import convolve
from numpy import random as rand
from kaggle_environments import evaluate, make, utils

class Board:
  def __init__(self, config, scoreFunc = False):
    self.config = config
    self.timeout = config["timeout"]
    self.columns = config["columns"]
    self.rows = config["rows"]
    self.inarow = config["inarow"]
    self.steps = config["steps"]
    if (scoreFunc):
      self.scoreFunc = scoreFunc
    else:
      self.scoreFunc = self.defaultScoreFunc

  # I think I might make the scorFunc more general, and move the decision to where I call this function.
  @staticmethod
  def defaultScoreFunc(outcomeDict, openBelow):
    return sum(np.sum((outcome > 0) * np.power(outcome, 3) / (openBelow[name] + 1)) for name, outcome in outcomeDict.items())
    # Options for scoreFunc:
    # return (max(np.amax(outcome) for name, outcome in outcomeDict.items()) / self.inarow)  ** self.SCOREPOWER
    # return sum(np.sum((outcome > 0) * np.power(outcome, self.SCOREPOWER)) for name, outcome in outcomeDict.items())

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
    return { "filtersOutCome": filtersOutCome, "openBelow": openBelow }

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
    myScore = self.scoreFunc(myState["filtersOutCome"], myState["openBelow"])
    
    opponentState = self.checkState(opponentBoard, myBoard, openBoard)
    opponentScore = self.scoreFunc(opponentState["filtersOutCome"], opponentState["openBelow"])
    
    score = myScore - opponentScore

    self.boardState = { 
      "score": score,
      "myBoard": myBoard,
      "opponentBoard": opponentBoard,
      "playOptions": playOptions,
      "myState": myState,
      "opponentState": opponentState
    }
    return self.boardState

  @staticmethod
  def test(printStates):
    observation = { 
      'board': [0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0,
                0, 2, 0, 0, 0, 0, 0,
                0, 1, 0, 2, 2, 1, 0,
                0, 2, 2, 1, 1, 1, 0,
                0, 2, 1, 1, 2, 1, 0],
      # 'board': [0, 0, 0, 0, 0, 0, 0,
      #           0, 0, 0, 0, 0, 0, 0,
      #           0, 0, 0, 0, 0, 0, 0,
      #           0, 0, 0, 0, 0, 0, 0,
      #           0, 0, 0, 0, 0, 0, 0,
      #           0, 0, 0, 0, 0, 0, 0],
      'mark': 1
    }

    configuration = {'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000}
    board = Board(configuration)
    state = board.assessState(observation)
    print("  BOARD STATUS:")
    print(1 * state["myBoard"] + 2 * state["opponentBoard"] + 3 * state["playOptions"])
    print("  MY SCORE: ", state["score"])
    if (printStates):
      print("\nMy state:")
      print("maximum value: ", state["myState"]["maxValue"])
      print("vertical:")
      print(state["myState"]["filtersOutCome"]["vertical"])
      print("horizontal:")
      print(state["myState"]["filtersOutCome"]["horizontal"])
      print("mainDiag: ")
      print(state["myState"]["filtersOutCome"]["mainDiag"])
      print("subDiag:")
      print(state["myState"]["filtersOutCome"]["subDiag"])

# Board.test(False)
