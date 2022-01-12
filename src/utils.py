import matplotlib.pyplot as plt

def plot_history(history, epochs, save_path=None, save_flg=False):
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), history['train_loss'], label='train_loss')
    ax.plot(range(1, epochs+1), history['validation_loss'], label='validation_loss')
    ax.set_title('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    if save_flg:
        plt.savefig(f'{save_path}/loss.png')
    else:
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), history['train_acc'], label='train_acc')
    ax.plot(range(1, epochs+1), history['validation_acc'], label='validation_acc')
    ax.set_title('accuracy')
    ax.set_xlabel('epoch')
    ax.legend()
    if save_flg:
        plt.savefig(f'{save_path}/test_acc.png')
    else:
        plt.show()
        
def init_history():
    return {
        'train_loss': [],
        'train_acc': [],
        'validation_loss': [],
        'validation_acc': [],
        'kappa_score': [],
    }   

def append_history(history: dict, train_loss, train_acc, val_loss, val_acc, score=None):
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc) 
    history['validation_loss'].append(val_loss)
    history['validation_acc'].append(val_acc)
    if score:
        history['kappa_score'].append(score)