import log_values
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


experiment = "your_experiment"
logger = SummaryWriter(comment=experiment)


def to_np(x):
    """
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    :param x:
    :return:
    """
    return x.data.cpu().numpy()

def train(train_loader, model, criterion, epoch, num_epochs):
    model.train()
    losses = log_values.AverageMeter()

    optimizer = optim.SGD(model.parameters(),
                          weight_decay=1e-4,
                          lr=1e-4,
                          momentum=0.9,
                          nesterov=True)

    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        # measure loss
        loss = criterion(outputs, labels)
        losses.update(loss.data, images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        #print('data/(train)loss_val', losses.val, i + 1)
        #print('data/(train)loss_avg', losses.avg, i + 1)

        # logging

        # add the model graph
        #logger.add_graph(model, outputs)

        # log loss values every iteration
        #logger.add_scalar('data/(train)loss_val', losses.val, i + 1)
        #logger.add_scalar('data/(train)loss_avg', losses.avg, i + 1)

        # log the layers and layers gradient histogram and distributions
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.add_histogram('model/(train)' + tag, to_np(value), i + 1)
        #     logger.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)
        #
        # # log the outputs given by the model (The segmentation)
        # logger.add_image('model/(train)output', make_grid(outputs.data), i + 1)

        # update progress bar status
        pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg) '
                             % (epoch + 1, num_epochs, losses.val, losses.avg))

    # return avg loss over the epoch
    return losses.avg