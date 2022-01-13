import torch
import numpy as np
import gc
import tqdm

from sklearn.metrics import cohen_kappa_score

def train(data_loader, model, scheduler, criterion, optimizer, device):
    # モデルを学習モードに
    model.train()
    # 損失を0で初期化
    train_loss = 0
    train_acc = 0   
    counter = 0
    # データローダ内のバッチについてのループ
    for data in tqdm.tqdm(data_loader, desc='train'):
        
        for param in model.parameters():
            param.grad = None
        
        # 画像と目的変数を持っている
        inputs = data['image']
        targets = data['targets']
        # デバイスに転送

        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)
        # オプティマイザの勾配を0で初期化
        optimizer.zero_grad()

        # モデルの学習
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # _, labels = torch.max(targets.data, 1)
        
        # 損失の計算
        # 多クラス分類なので nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        # 計算した損失の増加
        train_loss += loss
        
        # 逆誤差伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        # TODO:early stopの実装
        # TODO:learning rateの動的変更アルゴリズムの実装
        # スケジューラを使う場合、ここに処理を記述
        if scheduler is not None:
            scheduler.step()
        else:
            pass

        # 正解件数算出
        train_acc += (preds == targets).sum()
        counter += targets.size(0)

    total_train_loss = train_loss / len(data_loader)
    total_train_acc = train_acc / counter

    gc.collect()
    torch.cuda.empty_cache()
    del inputs, targets

    return total_train_loss, total_train_acc

def evaluate(data_loader, model, scheduler, criterion, device):
    """

    Args:
        data_loader ([type]): [description]
        model ([type]): [description]
        device ([type]): [description]
    """    
    """[summary]

    Returns:
        [type]: [description]
    """
    # モデルを評価モードに
    model.eval()

    # 損失を0で初期化
    eval_loss = 0
    eval_acc = 0
    
    total_preds = []
    total_labels = []
    

    
    counter = 0
    # メモリ節約のために勾配を計算しない
    with torch.no_grad():
    # with torch.inference_mode():
        for data in tqdm.tqdm(data_loader, desc='eval'):
            # 画像と目的変数を持っている
            inputs = data['image']
            targets = data['targets']

            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # _, labels = torch.max(targets.data, 1)
            
            total_preds.append(preds.to('cpu').numpy())
            # total_labels.append(labels.to('cpu').numpy())
            total_labels.append(targets.to('cpu').numpy())
            
            loss = criterion(outputs, targets)

            # 計算した損失の増加
            eval_loss += loss
            
            # 正解件数算出
            eval_acc += (preds == targets).sum()
            counter += targets.size(0)

    total_eval_loss = eval_loss / len(data_loader)
    total_eval_acc = eval_acc / counter

    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    
    # !カッパスコアの計算確認 -> OK
    score = cohen_kappa_score(total_labels, total_preds, weights="quadratic")

    if scheduler is not None:
        scheduler.step()
    else:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    del inputs, targets

    return total_eval_loss, total_eval_acc, score

# TODO:予測フェーズの実装
def test(data_loader, model, device, criterion):
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
    
    test_loss = 0
    test_acc = 0

    # 勾配を計算しない
    with torch.no_grad():
        for data in data_loader:
            inputs, targets = data
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            # モデルの予測
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 10))
            test_loss += loss.item()
            
            predict = outputs.argmax(dim=1, keepdim=True)
            test_acc += predict.eq(targets.view_as(predict)).sum().item()
            
            # 目的変数と予測をリストに変換
            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.detach().cpu().numpy().tolist()

            # リストに格納
            final_targets.extend(targets)
            final_outputs.extend(outputs)

    # 目的変数と予測と返す
    return final_outputs, final_targets