"""
Some helpful functions

"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
import PIL
from PIL import Image
import random


# ################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')

# number of parameters
def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

################################################################################
# torch dataset from numpy array
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize


    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)


# compute entropy of class labels; labels is a numpy array
def compute_entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    net = net.cuda()
    net.eval()

    n = len(images)
    if batch_size>n:
        batch_size=n
    dataset_pred = IMGs_dataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_labels_pred = np.zeros(n+batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).cuda()
            batch_size_curr = len(batch_images)

            outputs,_ = net(batch_images)
            _, batch_class_labels_pred = torch.max(outputs.data, 1)
            class_labels_pred[nimgs_got:(nimgs_got+batch_size_curr)] = batch_class_labels_pred.detach().cpu().numpy().reshape(-1)

            nimgs_got += batch_size_curr
            if verbose:
                pb.update((float(nimgs_got)/n)*100)
        #end for batch_idx
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred


#######################################################
## transformation on images

## horizontal flip images
def random_hflip(batch_images, return_flipped_indx=False):
    ''' for numpy arrays '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
    if return_flipped_indx:
        return batch_images, indx_gt
    else:
        return batch_images

def random_hflip_tensor(batch_images):
    ''' for torch tensors '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
    return batch_images

## normalize images
def normalize_images(batch_images, to_neg_one_to_one=True):
    batch_images = batch_images/255.0 #to [0,1]
    if to_neg_one_to_one:
        batch_images = (batch_images - 0.5)/0.5 
    return batch_images


## vertical flip images
def random_vflip(images, p=0.5):
    flip_mask = np.random.rand(images.shape[0]) < p
    flipped_images = np.where(flip_mask[:, None, None, None], images[:, :, ::-1, :], images)
    return flipped_images

## random rotation
def random_rotate_90_degrees(image):
    angle = random.choice([0, 90, 180, 270])
    if angle == 0:
        return image
    elif angle == 90:
        return np.rot90(image, k=1, axes=(1, 2))
    elif angle == 180:
        return np.rot90(image, k=2, axes=(1, 2))
    elif angle == 270:
        return np.rot90(image, k=3, axes=(1, 2))
 
def random_rotate(images):
    images = np.concatenate([random_rotate_90_degrees(image) for image in images], axis=0)
    return images[:,np.newaxis,:,:]



###########################################
# helper functions

def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

def check_unnormalized_imgs(img):
    """To check whether all elements of a Tensor are within the range [0, 255]"""
    # Convert to tensor if numpy array is provided
    if isinstance(img, np.ndarray):
        tensor = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        tensor = img
    else:
        raise TypeError("Input must be either PyTorch tensor or NumPy array")
    cond1 = torch.all((tensor >= 0) & (tensor <= 255)).item()
    cond2 = not torch.all((tensor >= 0) & (tensor <= 1)).item()
    return cond1 and cond2



###########################################
# extra functions




