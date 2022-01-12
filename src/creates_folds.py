import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

def stratified_kfold(df, n_split=5):
    # kfoldという新しい列を作り、-1で初期化
    df['kfold'] = -1
    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)
    
    # 目的変数を取り出す
    y = df.target.values
    
    # StratifiedKFoldクラスの初期化
    kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
    # kfold列を埋める
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        print(f'FOLD{fold}')
        df.loc[val_, 'kfold'] = fold
        
    # データセットを新しい列と共に保存
    df.to_csv('../input/train_folds.csv', index=False)

def kfold(df, n_split=5):    
    # kfoldという新しい列を作り、-1で初期化
    df['kfold'] = -1
    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)
    
    # 目的変数を取り出す
    y = df.target.values
    
    # KFoldクラスの初期化
    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    # kfold列を埋める
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
        
    # データセットを新しい列と共に保存
    df.to_csv('../input/train_folds.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('../input/train.csv')
    stratified_kfold(df)