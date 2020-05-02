from kaggle_environments import evaluate, make, utils
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
import copy
import json
from scipy import signal as spSig
from datetime import datetime

from deepAgent import ScoreModel
from deepAgent import DeepAgent
from board import Board
from agent import Agent

class ScoreTrainer:
  def __init__(self):
    #constants
    self.NUM_ITERATIONS = 2000
    self.EVAL_NUM = 100 # self.NUM_ITERATIONS / 20
    self.NUM_EPOCHS = 10
    self.BATCH_SIZE = 500
    self.LOOK_AHEAD = 3
    self.SAVE_PATH = "../models/score_training_lookAhead_" + str(self.LOOK_AHEAD) + "_"
    self.config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    self.NEW_GAME = { 
      'board': [0] * self.config["columns"] * self.config["rows"],
      'mark': 1
    }
    self.LEARNING_RATE = 0.01
    self.maxMinScores()
    #variables
    self.board = Board(self.config, self.scoreFunc)
    self.scoreModel = ScoreModel(self.config, self.LEARNING_RATE)
    self.agent = DeepAgent(self.scoreModel)

  @staticmethod
  def scoreFunc(outcomeDict, openBelow):
    # if(max(outcome) == self.config["inarow"]): return self.MAX_SCORE
    return sum(np.sum((outcome > 0) * np.power(outcome, 3) / (openBelow[name] + 1)) for name, outcome in outcomeDict.items())
    # Options for scoreFunc:
    # return (max(np.amax(outcome) for name, outcome in outcomeDict.items()) / self.inarow)  ** self.SCOREPOWER
    # return sum(np.sum((outcome > 0) * np.power(outcome, self.SCOREPOWER)) for name, outcome in outcomeDict.items())

  def maxMinScores(self):
    observation = {
      'board': [1] * self.config["columns"] * self.config["rows"],
      'mark': 1
    }
    board = Board(self.config)
    boardState = board.makeBoard(observation)
    filtersOutCome = board.filtersOutCome(boardState["myBoard"], boardState["opponentBoard"])
    openBelow = board.openBelow(boardState["openBoard"], filtersOutCome)
    self.MAX_SCORE = self.scoreFunc(filtersOutCome, openBelow)



  def getNextMove(self, observation, lookAhead = None):
    if lookAhead is None: lookAhead = self.LOOK_AHEAD
    currentState = self.board.assessState(observation)
    scoreList = -self.MAX_SCORE * np.ones(self.board.columns)
    for playOption in np.transpose(np.where(currentState["playOptions"])):
      observationCopy = copy.deepcopy(observation)
      observationCopy["board"][playOption[0] * self.board.columns + playOption[1]] = observation["mark"]
      if (lookAhead == 0):
        newState = self.board.assessState(observationCopy)
        scoreList[playOption[1]] = (newState["score"])
      else:
        observationCopy["mark"] = 3 - observation["mark"]
        scoreList[playOption[1]] = - np.max(self.getNextMove(observationCopy, lookAhead - 1))
    return scoreList

  def checkWin(self, observation):
    board = self.board.makeBoard(observation)
    filtersOutCome = self.board.filtersOutCome(board["myBoard"], board["opponentBoard"])
    myMaxInaRow = max(np.max(outcome[:]) for name, outcome in filtersOutCome.items())
    if myMaxInaRow == self.board.inarow: return 1
    filtersOutCome = self.board.filtersOutCome(board["opponentBoard"], board["myBoard"])
    hisMaxInaRow = max(np.max(outcome[:]) for name, outcome in filtersOutCome.items())
    if hisMaxInaRow == self.board.inarow: return -1
    if not board["playOptions"].any(): return -1
    return 0
    
  def makeBoard(self, observation):
    board = self.board.makeBoard(observation)
    boardMatrix = np.zeros((1, 3, self.config["rows"], self.config["columns"]))
    boardMatrix[0, 0,:,:] = board["myBoard"]
    boardMatrix[0, 1,:,:] = board["opponentBoard"]
    boardMatrix[0, 2,:,:] = board["playOptions"]
    return boardMatrix

  def playTurn(self, observation, modelOutput):
    board = self.board.makeBoard(observation)
    playableColumns = (np.sum(board["playOptions"], 0))
    modelOutput = [modelOutput[i] if playableColumns[i] else -self.MAX_SCORE for i in range(self.config["columns"])]
    playColumn = np.argmax(modelOutput)
    playRow = np.squeeze(np.where(board["playOptions"][:, playColumn]))
    observation["board"][playRow * self.config["columns"] + playColumn] = observation["mark"]
    observation["mark"] = 3 - observation["mark"]
    return observation
  
  def trainIteration(self):
    trainBoards = []
    groundTruth = []
    loss = []
    observation = copy.deepcopy(self.NEW_GAME)
    while self.checkWin(observation) == 0:
      boardTensor = self.makeBoard(observation)
      trainBoards.append(np.squeeze(boardTensor))
      groundTruth.append(self.getNextMove(observation))
      modelOutput = self.scoreModel.play(boardTensor)[0]
      observation = self.playTurn(observation, modelOutput)
    # print("board:\n", np.reshape(observation["board"], (6, 7)))
    loss.append(self.scoreModel.back(trainBoards, groundTruth))
    return { "trainBoards": trainBoards, "groundTruth": groundTruth, "loss": loss }

  def trainEpoch(self):
    trainBoards = []
    groundTruth = []
    loss = []
    for i in range(self.NUM_ITERATIONS):
      # print(datetime.now(), "- training iteration #%d:" % i)
      iterationData = self.trainIteration()
      trainBoards.extend(iterationData["trainBoards"])
      groundTruth.extend(iterationData["groundTruth"])
      loss.extend(iterationData["loss"])
      if (i + 1) % self.EVAL_NUM == 0:
        print(datetime.now())
        print(">>Iteration #%d:" % i)
        print(">>  Loss:", loss[-1])
        self.evaluate()
        self.scoreModel.save(self.SAVE_PATH + str(i) + ".pt")
    writeObj = json.dumps({ "trainBoards": trainBoards, "groundTruth": groundTruth, "loss": loss })
    file = open("../data/score_training_lookAhead_" + str(self.LOOK_AHEAD) + "_" + datetime.now())
    file.write(writeObj)
    file.close()
    
    print("\n>>Running retrain")
    epochLoss = []
    for i in range(self.NUM_EPOCHS):
      perm = np.random.permutation(range(len(groundTruth)))
      for i in range(int(len(perm) / 500)):
        currIndices = perm[500 * i : 500 * (i + 1)]
        tb = [trainBoards[i] for i in currIndices]
        gt = [groundTruth[i] for i in currIndices]
        epochLoss.append(self.scoreModel.back(tb, gt))
        print(">>  Loss:", epochLoss[-1])
      self.evaluate()
    plt.scatter(range(len(epochLoss)), epochLoss)
    return { "trainBoards": trainBoards, "groundTruth": groundTruth, "loss": loss }

  @staticmethod
  def match(agent0, agent1):
    rewards0 = evaluate("connectx", [agent0, agent1], num_episodes=5)
    rewards1 = evaluate("connectx", [agent1, agent0], num_episodes=5)
    out0 = sum(r[0] for r in rewards0) / sum(r[0] + r[1] for r in rewards0)
    out1 = sum(r[1] for r in rewards1) / sum(r[0] + r[1] for r in rewards1)
    out = 0.5 * (out0 + out1)
    return out

  def evaluate(self):
    agent = Agent(self.LOOK_AHEAD)
    print(">>  winner scores:")
    try:
      print(">>  random:", self.match(self.agent.play, "random"))
      print(">>  negamax:", self.match(self.agent.play, "negamax"))
      print(">>  trainer:", self.match(self.agent.play, agent.play))
    except:
      print("match returned error")


  def train(self):
    trainBoards = []
    groundTruth = []
    loss = []
    for i in range(1):
      epochData = self.trainEpoch()
      trainBoards.extend(epochData["trainBoards"])
      groundTruth.extend(epochData["groundTruth"])
      loss.extend(epochData["loss"])
      self.LOOK_AHEAD += 2
      self.NUM_ITERATIONS = int(self.NUM_ITERATIONS / 2)
      self.EVAL_NUM = int(self.EVAL_NUM / 2)
      self.SAVE_PATH = "../models/score_training_lookAhead_" + str(self.LOOK_AHEAD) + "_"

  def randomTrain(self):
    trainBoards = []
    groundTruth = []
    loss = []
    for i in range(1000):
      print("iteration", i)
      observation = copy.deepcopy(self.NEW_GAME)
      unplayed = self.config["columns"] * [self.config["rows"]]
      while self.checkWin(observation) == 0:
        boardTensor = self.makeBoard(observation)
        trainBoards.append(np.squeeze(boardTensor))
        groundTruth.append(self.getNextMove(observation))
        move = rand.randint(self.config["columns"])
        while unplayed[int(move)] == 0:
          move = rand.randint(self.config["columns"])

        unplayed[int(move)] -= 1
        play = [0] * self.config["columns"]
        play[int(move)] = 1
        observation = self.playTurn(observation, play)

    for i in range(self.NUM_EPOCHS):
      perm = np.random.permutation(range(len(groundTruth)))
      self.SAVE_PATH = "../models/score_training_random_epoch_"
      for i in range(int(len(perm) / self.BATCH_SIZE)):
        currIndices = perm[self.BATCH_SIZE * i : self.BATCH_SIZE * (i + 1)]
        tb = [trainBoards[i] for i in currIndices]
        gt = [groundTruth[i] for i in currIndices]
        loss.append(self.scoreModel.back(tb, gt))
      # print(">>  Loss:", loss[-1])
      self.scoreModel.save(self.SAVE_PATH + str(i) + ".pt")
    plt.scatter(range(len(loss)), loss)


  @staticmethod
  def test():
    trainer = ScoreTrainer()
    for i in range(2):
      trainer.trainIteration()

if __name__ == '__main__':
  print("\n\nScore Trainer Training session")
  trainer = ScoreTrainer()
  # trainer.randomTrain()
  trainer.train()
