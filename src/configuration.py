import os
import sys
import hydra
import pandas as pd
import sklearn.model_selection as sms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision 

sys.path.insert(0, f'{os.getcwd()}/src')
import dataset
import criterion

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_split(config: dict):
    """[summary]

    Args:
        config (dict): [description]

    Returns:
        [type]: [description]
    """    
    split_config = config['split']
    name = split_config['name']
    return sms.__getattribute__(name)(**split_config['params'])

def get_metadata(config: dict, phase='train'):
    """[summary]

    Args:
        config (dict): [description]
        phase (str, optional): [description]. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    original_path = hydra.utils.get_original_cwd()
    # hydraが走ってるとこのpath取得
    data_config = config['data']
    
    # NOTE:下記ならTrue/Falseでもいいかも。
    if phase == 'train':
        path_csv = f"{original_path}/{data_config['train_df_path']}"
        path_data = f"{original_path}/{data_config['train_data_dir']}"
    elif phase == 'test':
        path_csv = f"{original_path}/{data_config['test_df_path']}"
        path_data = f"{original_path}/{data_config['test_data_dir']}"
    else:
        raise NameError(f"phase:({phase}) must be 'train' or 'test'.")

    df = pd.read_csv(path_csv)
    return df, path_data

def get_loader(df: pd.DataFrame, datadir: str, config: dict, phase: str):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        datadir (str): [description]
        config (dict): [description]
        phase (str): [description]

    Returns:
        loader (variable): [description]
    """    
    dataset_config = config['dataset']
    name = dataset_config['name']
    loader_config = config['loader'][phase]
    dataset = datasets.__getattribute__(name)(
            df=df,
            datadir=datadir,
            phase=phase,
            config=dataset_config['params'],
            )
    loader = data.DataLoader(dataset, **loader_config)
    return loader

def get_criterion(config: dict):
    """[summary]

    Args:
        config (dict): [description]

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """    
    loss_config = config['loss']
    loss_name = loss_config['name']
    loss_params = loss_config['params']
    if (loss_params is None) or (loss_params == ''):
        loss_params = {}

    if hasattr(nn, loss_name): # torch.nnに同名のloss関数があったら
        criterion_ = nn.__getattribute__(loss_name)(**loss_params)
    else: # ない場合は、自作のcriterion moduleから持ってくる
        criterion_cls = criterion.__getattribute__(loss_name) # getattrで同名クラスを所得して（インスタンス化はまだ）
        
        # NOTE: if criterion_cls:でもいいのでは？
        if criterion_cls is not None:
            criterion_ = criterion_cls(**loss_params) # パラメータ渡してインスタンス化
        else:
            raise NotImplementedError()

    return criterion_

def get_optimizer(model: nn.Module, config: dict):
    """[summary]

    Args:
        model (nn.Module): [description]
        config (dict): [description]

    Returns:
        [type]: [description]
    """    
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config.get('name')

    return optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_config['params'])

def get_scheduler(optimizer, config: dict):
    """[summary]

    Args:
        optimizer ([type]): [description]
        config (dict): [description]

    Returns:
        [type]: [description]
    """    
    scheduler_config = config['scheduler']
    scheduler_name = scheduler_config.get('name')

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(optimizer, **scheduler_config['params'])
