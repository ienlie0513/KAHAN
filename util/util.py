import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
from tqdm.notebook import tqdm as nbtqdm
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

class Progressor:
    '''
        This class is for costumize tqdm bar, specify bar total, print train acc, test acc and loss after bar
        calculate and print train acc, test acc and loss every updata
    '''
    def __init__(self, mode, total, log=None):
        self.mode = mode
        self.total = total
        self.log = log
        
        if(mode == 'py'):
            self.progress = tqdm(total = total)
        if(mode == 'nb'):
            self.progress = nbtqdm(total = total)
        self.progress.set_description('0 ')
            
        self.train_loss_total = 0 
        self.test_loss_total = 0  
        self.train_acc_total = 0
        self.test_acc_total = 0
        
        # count tqdm progress
        self.count = 0
        
    def update(self, train_loss, test_loss, train_acc, test_acc, epoch):
        # update value
        self.train_loss_total += train_loss
        self.test_loss_total += test_loss
        self.train_acc_total += train_acc
        self.test_acc_total += test_acc
        
        self.count += 1
        
        # update tqdm
        self.progress.set_postfix(train_loss = self.train_loss_total/self.count, 
                        test_loss = self.test_loss_total/self.count, 
                        train_acc = self.train_acc_total/self.count, 
                        test_acc = self.test_acc_total/self.count)
        self.progress.update(1)

        self.log.write('epoch: {}, train acc: {:.4f}, train loss: {:.4f}, valid acc: {:.4f}, test loss: {:.4f}\n'.format(
            epoch, self.train_acc_total/self.count, self.train_loss_total/self.count, self.test_acc_total/self.count, self.test_loss_total/self.count))
        
    def reset(self, epoch):
        if(self.mode == 'py'):
            self.progress = tqdm(total = self.total)
        if(self.mode == 'nb'):
            self.progress = nbtqdm(total = self.total)
        self.progress.set_description('%d '%(int((epoch+1)/self.total)))
        self.count = 0

        self.train_loss_total = 0
        self.test_loss_total = 0
        self.train_acc_total = 0
        self.test_acc_total = 0
        self.log.flush()

# show train acc, test acc and loss trend
def show_result(train_acc, valid_acc, train_loss, valid_loss, save=None):
    plt.figure(figsize=(12, 8), dpi=1200)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(train_acc, label = 'train acc', linewidth=2)
    plt.plot(valid_acc, label = 'valid acc', linewidth=2)
    plt.legend()
    if save:
        plt.savefig('{}/{}_acc_{}.png'.format(save[1], save[2], save[0]), bbox_inches='tight')  # save as svg, add bbox_inches='tight'
    else:
        plt.show()

    plt.figure(figsize=(12, 8), dpi=1200)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(train_loss, label = 'train loss', linewidth=2)
    plt.plot(valid_loss, label = 'valid loss', linewidth=2)
    plt.legend()
    if save:
        plt.savefig('{}/{}_loss_{}.png'.format(save[1], save[2], save[0]), bbox_inches='tight')  # save as svg, add bbox_inches='tight'
    else:
        plt.show()

# calculate confusion matrix and visualize
def plot_confusion_matrix(y_true, y_pred, num_class, normalize=False, save=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    else:
        cm = cm.astype('float')/cm.sum()

    plt.figure(figsize=(12, 8), dpi=1200)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix', fontsize=22)
    plt.colorbar()

    plt.xticks(np.arange(num_class), fontsize=18, rotation=45)
    plt.yticks(np.arange(num_class), fontsize=18)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'black' if cm[i, j] < 0.4 else 'w'  # Change text color based on value in confusion matrix
            plt.text(x=j, y=i, s=('%.2f'%cm[i][j]), va='center', ha='center', color=color, fontsize=25)
    
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    if save: 
        plt.savefig('{}/{}_cm_{}.png'.format(save[1], save[2], save[0]), bbox_inches='tight')  # save as svg, add bbox_inches='tight'
    else:
        plt.show()


# calculate precision, recall and f1-score
def calculate_metrics(acc, targets, predicts, log=None):
    print ()
        
    precision = precision_score(targets, predicts, average='macro', zero_division=0)
    recall = recall_score(targets, predicts, average='macro')
    microf1 = f1_score(targets, predicts, average='micro')
    macrof1 = f1_score(targets, predicts, average='macro')

    print ('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}'.format(acc, precision, recall, microf1, macrof1))
    log.write('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, micro f1: {:.4f}, macro f1: {:.4f}\n'.format(acc, precision, recall, microf1, macrof1))

    return acc, precision, recall, microf1, macrof1