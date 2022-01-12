import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        model [type]: [description]
    """    
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    model = eval(model_name)(model_params)
    # eval関数　：　文字列をpythonのコードとして実行する
    # modelのインスタンス化してることになる
    return model

# ネットワーク構造の定義
class MyNet(nn.Module):
    def __init__(self, params=None):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 12*12*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 全結合層
        self.fc1 = nn.Linear(
            in_features=9216,
            out_features=4096
        )
        # 0.5は、50%のみを次の層に伝えるという意味
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(
            in_features=4096,
            out_features=4096
        )
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(
            in_features=4096,
            out_features=1000
        )

    def forward(self, image):
        # 画像のバッチからバッチサイズ、チャンネル、高さ、幅を取得
        # 元のサイズ：(bs, 3, 227, 227)
        bs, c, h, w = image.size()
        x = F.relu(self.conv1(image))   # サイズ： (bs, 96, 55, 55)
        x = self.pool1(x)           # サイズ： (bs, 96, 27, 27)
        x = F.relu(self.conv2(x))   # サイズ： (bs, 256, 13, 13)
        x = self.pool2(x)           # サイズ： (bs, 256, 13, 13)
        x = F.relu(self.conv3(x))   # サイズ： (bs, 384, 13, 13)
        x = F.relu(self.conv4(x))   # サイズ： (bs, 256, 13, 13)
        x = F.relu(self.conv5(x))   # サイズ： (bs, 256, 13, 13)
        x = self.pool3(x)           # サイズ： (bs, 256, 6, 6)
        x = x.view(bs, -1)          # サイズ： (bs, 9216)
        x = F.relu(self.fc1(x))     # サイズ： (bs, 4096)
        x = self.dropout1(x)        # サイズ： (bs, 4096)
        # 正規化に使うドロップアウト層ではサイズは不変
        x = F.relu(self.fc2(x))     # サイズ： (bs, 4096)
        x = self.dropout2(x)        # サイズ： (bs, 4096)
        x = F.relu(self.fc3(x))     # サイズ： (bs, 1000)
        # ImageNet データセットのカテゴリ数は 1000
        # 活性化関数のソフトマックスは、線形層（全結合層）の出力を
        # バッチ単位で1以下の確率に変換
        x = torch.softmax(x, dim=1)    # サイズ： (bs, 1000)
        return x
    