import torch
import build_unet_architecture
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import log_values
import numpy as np
import matplotlib.pyplot as plt
import loss
import os
from collections import OrderedDict

path = 'model_best_bce.pth.tar'
device = torch.device('cuda')
model = build_unet_architecture.CleanU_Net()
model.to(device)
model.eval()
state = torch.load(path)

# load params
model.load_state_dict(state['state_dict'])
#criterion = torch.nn.Sigmoid()
criterion = loss.BCELoss2d().cuda()
#loss.BCELoss2d().cuda()

def test(val_loader):


    losses = log_values.AverageMeter()

    optimizer = optim.SGD(model.parameters(),
                          weight_decay=1e-4,
                          lr=1e-4,
                          momentum=0.9,
                          nesterov=True)

    # set a progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    id = 0

    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        for index in range(0,len(images)):
            file_path = 'validation\\input_image_' + str(id)
            os.mkdir(file_path)

            image_ = np.squeeze(images[index].cpu().data.numpy())

            plt.imsave(file_path + '\\original_image.png',image_,cmap='gray')

            label_ = np.squeeze(labels[index].cpu().data.numpy())
            plt.imsave(file_path + '\\original_seg_output.png', label_,cmap='gray')

            pred_ = torch.sigmoid(outputs[index].cpu())
            pred_ = torch.Tensor(pred_)
            pred_ = np.squeeze(pred_)

            print(pred_.min(), pred_.max())
            # pred_ = torch.sigmoid(outputs[index].cpu())
            # pred_ = np.squeeze(pred_.data.numpy())
            #pred_ = np.squeeze(outputs[index].cpu().data.numpy())
            plt.imsave(file_path + '\\predicted_seg_output.png', pred_,cmap='gray')

            id += 1

        # measure loss
        loss = criterion(outputs, labels)
        losses.update(loss.data, images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        print('data/(train)loss_val', losses.val, i + 1)
        print('data/(train)loss_avg', losses.avg, i + 1)

        pbar.set_description('[TEST] - BATCH LOSS: %.4f/ %.4f(avg) '
                             % (losses.val, losses.avg))