import torch.nn as nn
# from torch.nn.modules.dropout import Dropout
# from torch.nn.modules.linear import Linear
import pretrainedmodels


def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__['alexnet'](
            pretrained = 'imagenet'
        )
    else:
        model = pretrainedmodels.__dict__['alexnet'](
            pretrained = None
        )

    # モデルを出力すると中身がわかる
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1),
    )
    return model

if __name__ == '__main__':
    print(get_model(pretrained=True))