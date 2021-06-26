import torch
import numpy as np

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
            correct_pixels = torch.sum(diff <= 1e-2)
            acc = correct_pixels.double() / (step_size*res)
            train_loss.update(loss, step_size)
            train_acc.updatee(acc, step_size)

            if epoch_step % (len(train_loader)/10) == 0 or epoch_step <= 10:
                print('Train epoch: [{}] step [{}/{}] loss: {:.4e} acc: {:.3f}'.format(epoch, epoch_step, len(train_loader), train_loss.avg, train_acc.avg))

        scheduler.step(train_loss.avg)

    return train_loss.avg, train_acc.avg


def validate():
    pass


def predict():
    pass


def plot_curves():
    pass