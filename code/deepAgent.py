import numpy as np
from numpy import random as rand
from scipy import signal as spSig
import torch
import torch.nn as nn
import torch.optim as optim

class ScoreModel(nn.Module):
  def __init__(self, config, LearningRate = 0.01):
    super(ScoreModel, self).__init__()
    self.config = config
    self.criterion = nn.MSELoss()
    self.conv1 = nn.Conv2d(3, 64, config["inarow"])  # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    self.conv2 = nn.Conv2d(64, 128, 3)
    self.createDenseLayers()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.optimizer = optim.SGD(self.parameters(), lr=LearningRate)

  # calculates the input size of the dense layer:
  def createDenseLayers(self):
    dummyIn = torch.randn(1, 3, self.config["rows"], self.config["columns"])
    dummyIn = nn.functional.relu(self.conv1(dummyIn))
    dummyOut = nn.functional.relu(self.conv2(dummyIn))
    self.denseInputSize = np.prod(dummyOut.shape)
    self.dense1 = nn.Linear(self.denseInputSize, 4 * self.config["columns"], bias="True")
    self.dense2 = nn.Linear(4 * self.config["columns"], self.config["columns"], bias="True")

  def forward(self, x):
    x = nn.functional.relu(self.conv1(x))
    x = nn.functional.relu(self.conv2(x))
    x = x.view(-1, self.denseInputSize)
    x = nn.functional.relu(self.dense1(x))
    x = nn.functional.normalize(self.dense2(x))
    return x

  def play(self, input):
    input = torch.from_numpy(input).float()
    if self.device.type == "cuda":
      input = input.to(self.device)
    output = self.forward(input)
    output = self.forward(input)
    if output.device.type == "cuda":
      output = output.to("cpu")
    return output.detach().numpy()

  def back(self, input, GT):
    input = torch.Tensor(input).float()
    if self.device.type == "cuda":
      input = input.to(self.device)
    self.zero_grad()
    output = self.forward(input)
    GT = (nn.functional.normalize(torch.Tensor(GT))).to(self.device)
    loss = self.criterion(output, GT)
    loss.backward()
    self.optimizer.step()
    return torch.Tensor.cpu(loss.data).numpy()

  def mutate(self, genNumber, GAMMA):
    modelLayers = [
      self.conv1,
      self.conv2,
      self.dense1
    ]

    for layer in modelLayers:
      layer.weight.data += (torch.randn(layer.weight.data.shape) * np.exp(-GAMMA * genNumber)).to(self.device)
      try:
        layer.bias.data += (torch.randn(layer.bias.data.shape) * np.exp(-GAMMA * genNumber)).to(self.device)
      except:
          pass
    return self

  def save(self, PATH):
    torch.save(self.state_dict(), PATH)

  @staticmethod
  def test():
    print("\nTesting the Score model")
    config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    model = ScoreModel(config)
    # test forward:
    dummyIn = rand.randn(1, 3, config["rows"], config["columns"])
    output = model.forward(dummyIn)
    print("\nModel output for randn input:")
    print(output)
    # test mutate:
    print("\nconv1 layer before mutation:")
    print(model.conv1.weight[0])
    model.mutate(1, 0.1)
    print("\nconv1 layer after mutation:")
    print(model.conv1.weight[0])
    return 0

class DeepAgent():
  def __init__(self, scoreModel):
    self.scoreModel = scoreModel
    if self.scoreModel.device.type == "cuda":
      self.scoreModel.to(self.scoreModel.device)

  def play(self, observation, config, *args):
    board = self.makeBoard(observation, config)
    playableColumns = np.sum(board[0, 2, :, :], 0)
    modelOutput = self.scoreModel.play(board)[0]
    validOutput = []
    for i in range(config["columns"]):
      if(playableColumns[i]):
        validOutput.append(modelOutput[i])
      else:
        validOutput.append(-np.inf)

    bestPlay = int(np.argmax(validOutput))
    return bestPlay

  def makeBoard(self, observation, config):
    board = np.reshape(observation["board"], (config["rows"], config["columns"]))
    myBoard = (board == observation["mark"])
    opponentBoard = (board == 3 - observation["mark"]) # so that if I'm 2 then they are 1 and the opposite
    openBoard = (board == 0)
    playOptions = self.playOptions(openBoard)
    board = np.zeros((1, 3, config["rows"], config["columns"]))
    board[0, 0,:,:] = myBoard
    board[0, 1,:,:] = opponentBoard
    board[0, 2,:,:] = playOptions
    return board

  def playOptions(self, openBoard):
    playOptionsFilter = np.array([[-1], [1]])
    playOptions = spSig.convolve2d(openBoard, playOptionsFilter, mode='full', boundary='fill', fillvalue=0)
    playOptions = (playOptions[1:, :] == 1)
    return playOptions

  @staticmethod
  def test():
    print("\nTesting the Deep Agent")
    observation = { 
    'board': [0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0,
              0, 2, 0, 0, 0, 0, 0,
              0, 1, 0, 2, 2, 1, 0,
              0, 2, 2, 1, 1, 1, 0,
              0, 2, 1, 1, 2, 1, 0],
    'mark': 1
    }
    config = { 'timeout': 5, 'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000 }
    scoreModel = ScoreModel(config)
    testAgent = DeepAgent(scoreModel)
    bestPlay = testAgent.play(observation, config)
    print("agents play: ", bestPlay)
    return 0

if __name__ == '__main__':
  test = ScoreModel.test()
  if test == 0:
    DeepAgent.test()
