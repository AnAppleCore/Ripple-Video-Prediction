import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

class Metrics(object):
    def __init__(self):
        self.val = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def train(train_loader, model, loss_func, optimizer, scheduler, epoch, args):

    train_loss = Metrics()
    train_acc = Metrics()
    res = args.img_shape[0]*args.img_shape[1]

    model.train()
    with torch.set_grad_enabled(True):
        for epoch_step, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.cuda()
            labels = labels.cuda()
            step_size = inputs.size(0) # current batch_size

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            # update statics
            loss = loss.item()
            diff = abs(outputs - labels)
            correct_pixels = torch.sum(diff <= 5e-2)
            acc = correct_pixels.double() / (step_size*res)
            train_loss.update(loss, step_size)
            train_acc.update(acc, step_size)

            # if epoch_step % (len(train_loader)/10) == 0 or epoch_step <= 10:
            #     print('Train epoch: [{}] step [{}/{}] loss: {:.4e} acc: {:.3f}'.format(epoch, epoch_step, len(train_loader), train_loss.avg, train_acc.avg))

        scheduler.step(train_loss.avg)

    return train_loss.avg, train_acc.avg


def validate(val_loader, model, loss_func, epoch, args):

    val_loss = Metrics()
    val_acc = Metrics()
    res = args.img_shape[0]*args.img_shape[1]

    model.eval()
    with torch.no_grad():
        for epoch_step, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            step_size = inputs.size(0) # current batch_size

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # update statistics
            loss = loss.item()
            diff = abs(outputs - labels)
            correct_pixels = torch.sum(diff <= 5e-2)
            acc = correct_pixels.double() / (step_size*res)
            val_loss.update(loss, step_size)
            val_acc.update(acc, step_size)

            # if epoch_step % (len(val_loader)/10) == 0 or epoch_step <= 10:
            #     print('Validate epoch: [{}] step [{}/{}] loss: {:.4e} acc: {:.3f}'.format(epoch, epoch_step, len(val_loader), val_loss.avg, val_acc.avg))

            # store some output for visualization
            #FIXME wether `detach()` here is necessary
            if epoch % (args.epochs/10) == 0 and epoch_step == 0:
                height = outputs[0].cpu().detach().numpy()
                height = height * 64 + 128
                cv2.imwrite(args.save_path+'/images/'+args.img_name.split('.')[0] +'/epoch_'+str(epoch)+'.png', height)
                if epoch == 0:
                    height = labels[0].cpu().detach().numpy()
                    height = height * 64 + 128
                    cv2.imwrite(args.save_path+'/images/'+args.img_name.split('.')[0] +'/label.png', height)

    return val_loss.avg, val_acc.avg

def predict():
    pass


def plot_curves(metrics, args):
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4
    plt.legend(lns, ['Train loss','Validation loss','Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig(args.save_path + '/learning_curve_0.png', bbox_inches='tight')