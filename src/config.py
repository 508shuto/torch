import torch
import torch.nn as nn
from src.model import MyNet


TRAINING_FILE = '../input/train'
MODEL_OUTPUT = '../models/'

SAVE_PATH = '../models/my_mnist_model.pth'
NUM_WORKERS = 8

# 学習設定
EPOCHS = 1
BATCH_SIZE = 512
MODEL = MyNet()
# OPTIMIZER = torch.optim.Adam()
CRITERION = nn.CrossEntropyLoss()