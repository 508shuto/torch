import os
import sys
import gc
import glob

import hydra
from omegaconf import DictConfig

import numpy as np
from omegaconf.dictconfig import DictConfig
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
import torchvision.transforms as transforms

from src.model import get_model
from src.dataset import load_MNIST
from src.engine import train, evaluate
from src.utils import plot_history
from src.configuration import get_device, get_optimizer, get_criterion, get_scheduler

# TODO:hydraの実装
# TODO:mlflowの実装
@hydra.main(config_path='./config', config_name='run.yaml')
def run(cfg: DictConfig) -> None:
    debug = cfg['globals']['debug']
    
    # エポック数
    if debug:
        epochs = 1
    else:
        epochs = cfg['globals']['num_epochs']
    batch_size = cfg['loader']['train']['batch_size']
    # 拡張用
    # val_batchsize = cfg['defaults']['loader']['valid']['batch_size']
    
    # GPUが使えるときは使う
    device = get_device()
    
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
    
    # TODO:datasetの呼び出しを実装
    dataset = torchvision.datasets.MNIST(root=hydra.utils.to_absolute_path('./data'),
                                    train=True,
                                    download=True,
                                    transform=transform)
    # TODO:kfoldの実装
    # NOTE:dataset宣言時にvalidationにもデータ拡張されてしまう
    # NOTE:foldはcsvだけで行って、hydraでmultirunするのがいいかも
    # 分割数
    if debug:
        n_splits = 2
    else:
        n_splits = 5
    shuffle = True
    random_state = 1234
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # データ数削減
    if debug:
        data_X = dataset.train_data[:1000]
        data_y = dataset.train_labels[:1000]
    else:
        data_X = dataset.train_data
        data_y = dataset.train_labels
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X=data_X, y=data_y)):
        print(f'fold {fold_idx}')

        # ネットワーク構造の構築
        model = get_model(cfg).to(device)

        # 最適化方法の設定
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg).to(device)
        # scheduler = get_scheduler(optimizer, cfg)
        
        # TODO:dataloaderの呼び出しを実装
        # MNISTデータのロード
        # data_loaders = load_MNIST(batch=batch_size, num_workers=cfg['loader']['train']['num_workers'])
        
        # TODO:Subsetの使い方調べる
        data_loaders = {
            'train': DataLoader(Subset(dataset, train_idx), shuffle=True, batch_size=batch_size),
            'validation': DataLoader(Subset(dataset, val_idx), shuffle=False, batch_size=batch_size)
        }

        # 学習結果の保存
        history = {
            'train_loss': [],
            'train_acc': [],
            'validation_loss': [],
            'validation_acc': [],
            'kappa_score': [],
        }    
        
        best_score = 0.
        best_thresh = 0.
        best_loss = np.inf
        
        # TODO:fast progressの実装
        for epoch in range(epochs): 
            print(f'epoch: {epoch} start')
            # 学習部分
            model.train() # 学習モード
            train_loss, train_acc = train(
                data_loader=data_loaders['train'],
                model=model,
                scheduler=None,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                ) 
            print(f'Training loss (ave.): {train_loss:.4f}, Accuracy: {train_acc:.4f}')

            # 検証部分
            val_loss, val_acc, score = evaluate(
                data_loader=data_loaders['validation'],
                model=model,
                scheduler=None,
                device=device,
                criterion=criterion,
                )
            print(f'Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Score: {score:.4f}')

            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc.item()) 
            history['validation_loss'].append(val_loss.item())
            history['validation_acc'].append(val_acc.item())
            history['kappa_score'].append(score)
            
            # モデル保存
            if debug:
                pass
            elif score > best_score:
                best_score = score
                print(f'Epoch: {epoch+1} - SaveBestScore: {best_score:.4f}')
                torch.save(model.state_dict(), hydra.utils.to_absolute_path(f'models/fold_{fold_idx:02d}_bestscore.pth'))

        # 結果表示
        print(history)
        if debug:
            pass
        else:
            plot_history(history, epochs)

if __name__ == '__main__':
    run()
