import torch
import shutil

# Create a function to save the model state
# def save_checkpoint(state, is_best, filename='checkpoints\\checkpoint.pth.tar'):
#     """
#     https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     :param state:
#     :param is_best:
#     :param filename:
#     :return:
#     """
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


def save_checkpoint(state, count, filename='checkpoints\\checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = 'checkpoints\\checkpoint_' + str(count) + '.pth.tar'
    torch.save(state, filename)
