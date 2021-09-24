import torch
from torch._C import dtype
import torch.nn as nn

from tqdm import tqdm

def train(data_loader, model, optimizer, device):
    """
    
    Args:
        data_loader ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        device ([type]): [description]
    """    

    # モデルを学習モードに
    model.train()

    # データローダ内のバッチについてのループ
    for data in data_loader:
        # 画像と目的変数を持っている
        inputs = data['image']
        targets = data['targets']
        
        # デバイスに転送
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        # オプティマイザの勾配を0で初期化
        optimizer.zero_grad()
        # モデルの学習
        outputs = model(inputs)
        # 損失の計算
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        # 逆誤差伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # スケジューラを使う場合、ここに処理を記述

def evaluate(data_loader, model, device):
    """

    Args:
        data_loader ([type]): [description]
        model ([type]): [description]
        device ([type]): [description]
    """    

    # モデルを評価モードに
    model.eval()

    # 目的変数と予測を格納するリスト
    final_targets = []
    final_outputs = []

    # 勾配を計算しない
    with torch.no_grad():

        for data in data_loader:
            inputs = data['image']
            targets = data['targets']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # モデルの予測
            output = model(inputs)

            # 目的変数と予測をリストに変換
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # リストに格納
            final_targets.extend(targets)
            final_outputs.extend(output)

    # 目的変数と予測と返す
    return final_outputs, final_targets