import numpy as np
from scipy import signal as spSig
from numpy import random as rand
from kaggle_environments import evaluate, make, utils

class Board:
  def __init__(self, config):
    self.config = config
    
  def assessState(self, observation):
    board = np.reshape(observation.board, (self.config.columns, self.config.rows))
    myState = (board == observation.mark)
    opponentState = (board == 3 - observation.mark) # so that if I'm 2 then they are 1 and the opposite
    openBoard = (board == 0)
    playOptions = spSig.convolve2d(openBoard, [[1], [-1]], mode='full', boundary='fill', fillvalue=0)

def test():
  observation = {'board': [0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 2, 2, 1, 0,
                           0, 2, 2, 1, 1, 1, 0,
                           0, 2, 1, 1, 2, 1, 0],
                  'mark': 1}
  configuration = {'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000}
  board = Board(configuration)
  board.assessState(observation)

test()