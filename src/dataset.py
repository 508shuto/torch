import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import hydra


from PIL import Image
from PIL import ImageFile

# 終了を示すビットを持たない（破損した）画像に対応するための処理
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    """
    
    """
    def __init__(self, image_path, targets, resize=None, augmentations=None):
        """
        Args:
            image_path ([type]): [description]
            targets ([type]): [description]
            resize ([type], optional): [description]. Defaults to None.
            augmentations ([type], optional): [description]. Defaults to None.
        """        
        self.image_paths = image_path
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        """
        
        """        
        return len(self.image_paths)

    def __getitem__(self, item):
        """

        Args:
            item ([type]): [description]
        """        
        # PILを使って画像を開く
        image = Image.open(self.image_paths[item])
        # グレースケールをRGBに変換
        image = image.convert('RGB')
        # 目的変数の準備
        targets = self.targets[item]

        # 画像の変形
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.BILINEAR
            )

        # numpy 配列に変換
        image = np.array(image)

        # albumentations によるデータ拡張
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        # PyTorch で期待される形式に変換
        # !（高さ、幅、チャンネル）でなく（チャンネル、高さ、幅）
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # 画像と目的変数のテンソルを返す
        # 型に注目
        # 回帰問題の場合は、目的変数の型が torch.float
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'targets': torch.tensor(targets, dtype=torch.long),
        }
        
# MNISTデータセットのロード
def load_MNIST(batch=128, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST(root=hydra.utils.to_absolute_path('./data'),
                                        train=True,
                                        download=True,
                                        transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch,
                                        shuffle=True,
                                        num_workers=num_workers)

    val_dataset = torchvision.datasets.MNIST(root=hydra.utils.to_absolute_path('./data'),
                                        train=False,
                                        download=True,
                                        transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=batch,
                                        shuffle=True,
                                        num_workers=num_workers)

    return {'train': train_loader, 'validation': val_loader}