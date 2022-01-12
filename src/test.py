import pandas as pd
import torch
import tqdm

def test(df, data_loader, model, device):
    df = pd.read_csv('../input/test_images.csv')
    tk_test = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    
    preds = []
    for i, (images) in tk_test:
        images = images.to(device)
        outputs = model(images)
        _, y_preds = torch.max(outputs, 1)
    
        preds.extend(y_preds.to('cpu').numpy())
    print(f'preds:{len(preds)}')
    
    df['preds'] = preds
    
    return df