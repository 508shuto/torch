import os

import numpy as np
import pandas as pd

from PIL import Image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(training_df, image_dir):
    """
    
    Args:
        training_df ([type]): [description]
        image_dir ([type]): [description]
    """    
    # 画像のベクトルを格納するリスト
    images = []
    # 目的変数を格納するリスト
    targets = []
    # それぞれのデータについてのループ
    for index, row in tqdm(
        training_df.iterrows(), 
        total=len(training_df),
        desc='processing images'
    ):
        # 画像インデックス
        image_id = row['ImageId']
        # 画像パス
        image_path = os.path.join(image_dir, image_id)
        # PILによる画像の読み込み
        image = Image.open(image_path + '.png')
        # 画像を256x256に変形。リサンプリングにはバイリニア法を指定。
        image = image.resize((256, 256), resample=Image.BILINEAR)
        # numpy 配列に変換
        image = np.array(image)
        # 平坦化
        image = image.ravel()
        # リストに格納
        images.append(image)
        targets.append(int(row['targets']))
    # numpy 配列に変換
    images = np.array(images)
    # 配列のサイズを表示
    print(images.shape)
    return images, targets

if __name__ == '__main__':
    csv_path = './input/train.csv'
    image_path = './input/train_png/'

    # 画像インデックスと目的変数の列を含むCSVファイルを読み込み
    df = pd.read_csv(csv_path)

    # kfoldという新しい列を作り、-1で初期化
    df['kfold'] = -1

    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    # 目的変数を取り出す
    y = df.targets.values

    # StratifiedKFold クラスの初期化
    kf = model_selection.StratifiedKFold(n_splits=5)

    # kfold列を埋める
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # 各分割についてのループ
    for fold_ in range(5):
        # 学習用と評価用に分割
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # 特徴量と目的変数に変換
        # 時間の節約のためにループの外で処理することも可能
        xtrain, ytrain = create_dataset(train_df, image_path)

        # 特徴量と目的変数に変換
        # 時間の節約のためにループの外で処理することも可能
        xtest, ytest = create_dataset(test_df, image_path)

        # 標準のパラメータでランダムフォレストモデルを学習
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)

        # クラス1の予測確率
        preds = clf.predict_proba(xtest)[:, 1]

        # 結果の表示
        print(f'FOLD: {fold_}')
        print(f'AUC: {metrics.roc_auc_score(ytest, preds)}')
        print('')