import numpy as np

def accuracy(y_true, y_pred):
    # 
    correct_counter = 0
    
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            # 
            correct_counter += 1
            
    return correct_counter / len(y_true)

