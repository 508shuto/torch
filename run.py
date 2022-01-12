from genericpath import exists
import os
import sys
import gc
import glob

import hydra
from omegaconf import DictConfig

import numpy as np
from omegaconf.dictconfig import DictConfig
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
import albumentations

from src.model import get_model
from src.dataset import load_MNIST, ClassificationDataset
from src.engine import train, evaluate
from src.utils import plot_history, init_history, append_history
from src.configuration import get_device, get_optimizer, get_criterion, get_scheduler

# TODO:hydraの実装
# TODO:mlflowの実装

def create_train_val_info(df, cfg):
    df_train = df[df.kfold != cfg['globals']['fold']].reset_index(drop=True)
    train_pth = df_train.ImageID.values.tolist()
    train_pth = [hydra.utils.to_absolute_path(os.path.join(cfg['data']['train_data_dir'], i)) for i in train_pth]
    train_targets = df_train.target.values

    df_valid = df[df.kfold == cfg['globals']['fold']].reset_index(drop=True)
    valid_pth = df_valid.ImageID.values.tolist()
    valid_pth = [hydra.utils.to_absolute_path(os.path.join(cfg['data']['train_data_dir'], i)) for i in valid_pth]
    valid_targets = df_valid.target.values    
    return train_pth, train_targets, valid_pth, valid_targets

@hydra.main(config_path='./config', config_name='run.yaml')
def run(cfg: DictConfig) -> None:
    debug = cfg['globals']['debug']
    fold = cfg['globals']['fold']
    
    df = pd.read_csv(hydra.utils.to_absolute_path(cfg['data']['train_df_path']))
    
    train_images, train_targets, valid_images, valid_targets = create_train_val_info(df, cfg)
    
    # GPUが使えるときは使う
    device = get_device()

    transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ]),
    }
    
    # ImageNet データセットの各チャンネルの平均と標準偏差
    # 事前学習済みの重みを使う場合、事前時計算した値を利用
    # 使わない場合、対象のデータセットで別途計算した値を使う
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.244, 0.225)

    # albumentations は様々な画像のデータ拡張が利用できるライブラリ
    # ここでは正規化のみを利用
    # always_apply=Trueにして、正規化を常に適用
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    # エポック数
    if debug:
        epochs = 1
        train_dataset = torchvision.datasets.MNIST(
            root=hydra.utils.to_absolute_path('./data'),
            train=True,
            download=True,
            transform=transform['train']
            )
        valid_dataset = torchvision.datasets.MNIST(
            root=hydra.utils.to_absolute_path('./data'),
            train=False,
            download=True,
            transform=transform['valid'],
            )        
    else:
        epochs = cfg['globals']['num_epochs']
        # batch_size = cfg['loader']['train']['batch_size']    
        # TODO:datasetの呼び出しを実装
        train_dataset = ClassificationDataset(
            image_path=train_images,
            targets=train_targets,
            resize=(28, 28),
            augmentations=aug
            )
        valid_dataset = ClassificationDataset(
            image_path=valid_images,
            targets=valid_targets,
            resize=(28, 28),
            augmentations=aug
            )

    
    # TODO:FOLD毎にデータを読み出すDataLoaderの作成
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=cfg['loader']['train']['batch_size'],
        num_workers=cfg['loader']['train']['num_workers'],
        )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        batch_size=cfg['loader']['valid']['batch_size'],
        num_workers=cfg['loader']['valid']['num_workers'],
        )
    
    # ネットワーク構造の構築
    model = get_model(cfg).to(device)

    # 最適化方法の設定
    optimizer = get_optimizer(model, cfg)
    criterion = get_criterion(cfg).to(device)
    # scheduler = get_scheduler(optimizer, cfg)

    # 学習結果の保存
    history = init_history()
    
    best_score = 0.
    best_thresh = 0.
    best_loss = np.inf
    early_epoch = 10
    count = 0
    
    # TODO:fast progressの実装
    for epoch in range(epochs): 
        print(f'epoch: {epoch} start')
        # 学習部分
        model.train() # 学習モード
        train_loss, train_acc = train(
            data_loader=train_loader,
            model=model,
            scheduler=None,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            ) 
        print(f'Training loss (ave.): {train_loss:.4f}, Accuracy: {train_acc:.4f}')

        # 検証部分
        val_loss, val_acc, score = evaluate(
            data_loader=valid_loader,
            model=model,
            scheduler=None,
            device=device,
            criterion=criterion,
            )
        print(f'Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Score: {score:.4f}')
        
        append_history(
            history,
            train_loss.item(),
            train_acc.item(),
            val_loss.item(),
            val_acc.item(),
            score,
            )
        # モデル保存
        if debug:
            pass
        elif score > best_score:
            count = 0
            best_score = score
            print(f'Epoch: {epoch+1} - SaveBestScore: {best_score:.4f}')
            if not os.path.exists(hydra.utils.to_absolute_path(cfg['data']['model_save_dir'])):
                os.makedirs(hydra.utils.to_absolute_path(cfg['data']['model_save_dir']), exist_ok=True)
            else:
                pass
            torch.save(model.state_dict(), hydra.utils.to_absolute_path(f'{cfg['data']['model_save_dir']}/fold_{fold:02d}_bestscore.pth'))
        elif early_epoch < count:
            break
        else:
            count += 1
            
    # 結果表示
    print(history)
    if debug:
        pass
    else:
        plot_history(history, epochs)
                
if __name__ == '__main__':
    run()