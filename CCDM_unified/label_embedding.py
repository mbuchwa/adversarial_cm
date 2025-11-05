import os
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import timeit
from einops import rearrange, reduce, repeat, pack, unpack

from CCDM_unified.models import ResNet34_embed_y2h, model_y2h, ResNet34_embed_y2cov, model_y2cov
from CCDM_unified.utils import IMGs_dataset



class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding conditioning vectors."""

    """ from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.scale = float(scale)
        # Lazily initialised so the projection matches the incoming label dimension.
        self.register_buffer("_proj_weight", torch.empty(0), persistent=False)

    def _build_projection(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        weight = torch.randn(input_dim, self.embed_dim // 2, device=device, dtype=dtype)
        weight *= self.scale
        self._proj_weight = weight

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        input_dim = x.shape[1]
        if self._proj_weight.numel() == 0 or self._proj_weight.shape[0] != input_dim:
            self._build_projection(input_dim, x.device, x.dtype)

        x_proj = 2 * np.pi * x @ self._proj_weight
        x_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        if x_emb.shape[-1] < self.embed_dim:
            pad_width = self.embed_dim - x_emb.shape[-1]
            x_emb = F.pad(x_emb, (0, pad_width))

        return x_emb.view(x.shape[0], self.embed_dim)


class LabelEmbed:
    def __init__(self, dataset, path_y2h, path_y2cov, y2h_type="sinusoidal", y2cov_type="sinusoidal", h_dim = 128, cov_dim = 64**2*3, batch_size=128, nc=3, device="cuda"):
        self.dataset = dataset
        self.path_y2h = path_y2h 
        self.path_y2cov = path_y2cov
        self.y2h_type = y2h_type
        self.y2cov_type = y2cov_type
        self.h_dim = h_dim 
        self.cov_dim = cov_dim
        self.batch_size = batch_size
        self.nc = nc
        self.label_shape = None
        self.device = device
        self._cached_train_data = None

        assert y2h_type in ['resnet', 'sinusoidal', 'gaussian']
        assert y2cov_type in ['resnet', 'sinusoidal', 'gaussian']

        if (y2h_type == "resnet" or y2cov_type == "resnet") and self.dataset is None:
            raise ValueError(
                "LabelEmbed with ResNet embeddings requires a dataset implementing load_train_data()"
            )

        def _prepare_train_data():
            if self._cached_train_data is not None:
                return self._cached_train_data
            if self.dataset is None:
                raise ValueError(
                    "LabelEmbed requires dataset to prepare training data"
                )

            train_images, _, train_labels = self.dataset.load_train_data()
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels, dtype=np.float32)
            train_labels = train_labels.reshape(train_labels.shape[0], -1)
            label_shape = (train_labels.shape[1],)
            reshaped_labels = train_labels.reshape(train_labels.shape[0], *label_shape)
            unique_labels_norm = np.unique(train_labels, axis=0).reshape(-1, *label_shape)

            self.label_shape = label_shape
            self._cached_train_data = (
                train_images,
                reshaped_labels,
                unique_labels_norm,
            )
            return self._cached_train_data

        ## if type is resnet, we need to train two networks for label embedding
        if y2h_type == "resnet":

            os.makedirs(path_y2h, exist_ok=True)

            ## training setups
            epochs_resnet = 10
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-2
            
            ## training dataset
            train_images, train_labels, unique_labels_norm = _prepare_train_data()
            trainset = IMGs_dataset(train_images, train_labels, normalize=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

            ## training embedding network for y2h
            resnet_y2h_filename_ckpt = os.path.join(self.path_y2h, 'ckpt_resnet_y2h_epoch_{}.pth'.format(epochs_resnet))
            mlp_y2h_filename_ckpt = os.path.join(self.path_y2h, 'ckpt_mlp_y2h_epoch_{}.pth'.format(epochs_mlp))
            
            # init network
            model_resnet_y2h = ResNet34_embed_y2h(dim_embed=self.h_dim, nc=self.nc, label_shape=self.label_shape)
            model_resnet_y2h = model_resnet_y2h.to(device)
            model_resnet_y2h = nn.DataParallel(model_resnet_y2h)
            model_mlp_y2h = model_y2h(dim_embed=self.h_dim, label_shape=self.label_shape)
            model_mlp_y2h = model_mlp_y2h.to(device)
            model_mlp_y2h = nn.DataParallel(model_mlp_y2h)
            
            # training or loading existing ckpt
            if not os.path.isfile(resnet_y2h_filename_ckpt):
                print("\n Start training CNN for y2h label embedding >>>")
                model_resnet_y2h = train_resnet(net=model_resnet_y2h, net_name="resnet_y2h", trainloader=trainloader, epochs=epochs_resnet, resume_epoch = 0, lr_base=base_lr_resnet, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = self.path_y2h, device=device, label_shape=self.label_shape)
                # save model
                torch.save({
                'net_state_dict': model_resnet_y2h.state_dict(),
                }, resnet_y2h_filename_ckpt)
            else:
                print("\n resnet_y2h ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(resnet_y2h_filename_ckpt, weights_only=True)
                model_resnet_y2h.load_state_dict(checkpoint['net_state_dict'])
            #end not os.path.isfile
            
            # training or loading existing ckpt
            if not os.path.isfile(mlp_y2h_filename_ckpt):
                print("\n Start training mlp_y2h >>>")
                model_h2y = model_resnet_y2h.module.h2y
                model_mlp_y2h = train_mlp(unique_labels_norm = unique_labels_norm, model_mlp=model_mlp_y2h, model_name="mlp_y2h", model_h2y=model_h2y, epochs=500, lr_base=base_lr_mlp, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128, device=device, label_shape=self.label_shape)
                # save model
                torch.save({
                'net_state_dict': model_mlp_y2h.state_dict(),
                }, mlp_y2h_filename_ckpt)
            else:
                print("\n model mlp_y2h ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(mlp_y2h_filename_ckpt, weights_only=True)
                model_mlp_y2h.load_state_dict(checkpoint['net_state_dict'])
            #end not os.path.isfile
            
            self.model_mlp_y2h = model_mlp_y2h
            
            ##some simple test
            indx_tmp = np.arange(len(unique_labels_norm))
            np.random.shuffle(indx_tmp)
            indx_tmp = indx_tmp[:10]
            labels_tmp = torch.from_numpy(unique_labels_norm[indx_tmp]).type(torch.float).to(device)
            epsilons_tmp = torch.randn_like(labels_tmp) * 0.2
            labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)
            model_resnet_y2h.eval()
            net_h2y = model_resnet_y2h.module.h2y
            model_mlp_y2h.eval()
            with torch.no_grad():
                labels_hidden_tmp = model_mlp_y2h(labels_tmp.reshape(labels_tmp.shape[0], -1))
                labels_noise_hidden_tmp = model_mlp_y2h(labels_noise_tmp.reshape(labels_noise_tmp.shape[0], -1))
                labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(labels_tmp.shape[0], -1)
                labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(labels_tmp.shape[0], -1)
                labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
                labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
            labels_tmp = labels_tmp.cpu().numpy().reshape(labels_rec_tmp.shape)
            labels_noise_tmp = labels_noise_tmp.cpu().numpy().reshape(labels_noise_rec_tmp.shape)
            results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
            print("\n labels vs reconstructed labels")
            print(results1)
            results2 = np.concatenate((labels_noise_tmp, labels_noise_rec_tmp), axis=1)
            print("\n noisy labels vs reconstructed labels")
            print(results2)
            
        ##end if

        if y2cov_type == "resnet":
            
            os.makedirs(path_y2cov, exist_ok=True)
            
            ## training setups
            epochs_resnet = 10
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-3
            
            ## training dataset
            train_images, train_labels, unique_labels_norm = _prepare_train_data()
            trainset = IMGs_dataset(train_images, train_labels, normalize=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            
            ## trainin embedding network for y2cov
            resnet_y2cov_filename_ckpt = os.path.join(self.path_y2cov, 'ckpt_resnet_y2cov_epoch_{}.pth'.format(epochs_resnet))
            mlp_y2cov_filename_ckpt = os.path.join(self.path_y2cov, 'ckpt_mlp_y2cov_epoch_{}.pth'.format(epochs_mlp))
            
            # init network
            model_resnet_y2cov = ResNet34_embed_y2cov(dim_embed=self.cov_dim, nc=self.nc, label_shape=self.label_shape)
            model_resnet_y2cov = model_resnet_y2cov.to(device)
            model_resnet_y2cov = nn.DataParallel(model_resnet_y2cov)
            model_mlp_y2cov = model_y2cov(dim_embed=self.cov_dim, label_shape=self.label_shape)
            model_mlp_y2cov = model_mlp_y2cov.to(device)
            model_mlp_y2cov = nn.DataParallel(model_mlp_y2cov)
            
            # training or loading existing ckpt
            if not os.path.isfile(resnet_y2cov_filename_ckpt):
                print("\n Start training CNN for y2cov label embedding >>>")
                model_resnet_y2cov = train_resnet(net=model_resnet_y2cov, net_name="resnet_y2cov", trainloader=trainloader, epochs=epochs_resnet, resume_epoch = 0, lr_base=base_lr_resnet, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = self.path_y2cov, device=device, label_shape=self.label_shape)
                # save model
                torch.save({
                'net_state_dict': model_resnet_y2cov.state_dict(),
                }, resnet_y2cov_filename_ckpt)
            else:
                print("\n resnet_y2cov ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(resnet_y2cov_filename_ckpt, weights_only=True)
                model_resnet_y2cov.load_state_dict(checkpoint['net_state_dict'])
            #end not os.path.isfile
            
            # training or loading existing ckpt
            if not os.path.isfile(mlp_y2cov_filename_ckpt):
                print("\n Start training mlp_y2cov >>>")
                model_h2y = model_resnet_y2cov.module.h2y
                model_mlp_y2cov = train_mlp(unique_labels_norm = unique_labels_norm, model_mlp=model_mlp_y2cov, model_name="mlp_y2cov", model_h2y=model_h2y, epochs=500, lr_base=base_lr_mlp, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128, device=device, label_shape=self.label_shape)
                # save model
                torch.save({
                'net_state_dict': model_mlp_y2cov.state_dict(),
                }, mlp_y2cov_filename_ckpt)
            else:
                print("\n model mlp_y2cov ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(mlp_y2cov_filename_ckpt, weights_only=True)
                model_mlp_y2cov.load_state_dict(checkpoint['net_state_dict'])
            #end not os.path.isfile
            
            self.model_mlp_y2cov = model_mlp_y2cov
            
            ##some simple test
            indx_tmp = np.arange(len(unique_labels_norm))
            np.random.shuffle(indx_tmp)
            indx_tmp = indx_tmp[:10]
            labels_tmp = torch.from_numpy(unique_labels_norm[indx_tmp]).type(torch.float).to(device)
            epsilons_tmp = torch.randn_like(labels_tmp) * 0.2
            labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)
            model_resnet_y2cov.eval()
            net_h2y = model_resnet_y2cov.module.h2y
            model_mlp_y2cov.eval()
            with torch.no_grad():
                labels_hidden_tmp = model_mlp_y2cov(labels_tmp.reshape(labels_tmp.shape[0], -1))
                labels_noise_hidden_tmp = model_mlp_y2cov(labels_noise_tmp.reshape(labels_noise_tmp.shape[0], -1))
                labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(labels_tmp.shape[0], -1)
                labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(labels_tmp.shape[0], -1)
                labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
                labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
            labels_tmp = labels_tmp.cpu().numpy().reshape(labels_rec_tmp.shape)
            labels_noise_tmp = labels_noise_tmp.cpu().numpy().reshape(labels_noise_rec_tmp.shape)
            results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
            print("\n labels vs reconstructed labels")
            print(results1)
            results2 = np.concatenate((labels_noise_tmp, labels_noise_rec_tmp), axis=1)
            print("\n noisy labels vs reconstructed labels")
            print(results2)
            
        ##end if
            
          
        
    ## function for y2h
    def _reshape_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.reshape(labels.shape[0], -1).float()
        return labels

    def _sinusoidal_embedding(self, labels: torch.Tensor, embed_dim: int, add_bias: bool) -> torch.Tensor:
        max_period = 10000
        labels = self._reshape_labels(labels)
        cond_dim = labels.shape[1]
        if embed_dim % cond_dim != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by conditioning dimension {cond_dim}"
            )

        per_dim = embed_dim // cond_dim
        if per_dim < 2:
            raise ValueError(
                "Each conditioning feature must receive at least a two-dimensional sinusoidal embedding"
            )
        half = per_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=labels.dtype, device=labels.device)
            / half
        )
        embeddings = []
        for dim_idx in range(cond_dim):
            args = labels[:, dim_idx : dim_idx + 1] * freqs
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if per_dim % 2:
                emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            embeddings.append(emb)

        embedding = torch.cat(embeddings, dim=-1)
        if add_bias:
            embedding = (embedding + 1) / 2
        return embedding

    def fn_y2h(self, labels):
        embed_dim = self.h_dim
        if self.y2h_type == "sinusoidal":
            embedding = self._sinusoidal_embedding(labels, embed_dim, add_bias=True)

        elif self.y2h_type == "gaussian":
            labels = self._reshape_labels(labels)
            embedding = GaussianFourierProjection(embed_dim=embed_dim)(labels)
            embedding = (embedding + 1) / 2  # make sure the embedding is not negative, and in [0,1]

        elif self.y2h_type == "resnet":
            self.model_mlp_y2h.eval()
            target_device = torch.device(self.device)
            self.model_mlp_y2h = self.model_mlp_y2h.to(target_device)
            reshaped_labels = self._reshape_labels(labels).to(target_device)
            embedding = self.model_mlp_y2h(reshaped_labels)

        return embedding
    
    ## function for y2cov
    def fn_y2cov(self, labels):
        embed_dim = self.cov_dim
        if self.y2cov_type == "sinusoidal":
            embedding = self._sinusoidal_embedding(labels, embed_dim, add_bias=False)
            embedding = embedding + 1  # make sure the embedding is not negative

        elif self.y2cov_type == "gaussian":
            labels = self._reshape_labels(labels)
            embedding = GaussianFourierProjection(embed_dim=embed_dim)(labels)
            embedding = embedding + 1  # make sure the embedding is not negative

        elif self.y2cov_type == "resnet":
            self.model_mlp_y2cov.eval()
            target_device = torch.device(self.device)
            self.model_mlp_y2cov = self.model_mlp_y2cov.to(target_device)
            reshaped_labels = self._reshape_labels(labels).to(target_device)
            embedding = self.model_mlp_y2cov(reshaped_labels)

        return embedding




def train_resnet(net, net_name, trainloader, epochs=200, resume_epoch = 0, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = None, device="cuda", label_shape=(1,)):
    
    ''' learning rate decay '''
    def adjust_learning_rate_1(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer_resnet = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    # resume training; load checkpoint
    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(net_name, net_name, resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer_resnet.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    if label_shape is None:
        label_shape = (1,)
    elif isinstance(label_shape, (list, tuple)):
        label_shape = tuple(label_shape)
    else:
        label_shape = (label_shape,)
    label_dim = int(np.prod(label_shape)) if label_shape else 1

    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_1(optimizer_resnet, epoch)
        for _, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_size_curr = batch_train_labels.shape[0]
            batch_train_labels = batch_train_labels.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.reshape(batch_size_curr, label_dim)

            #Forward pass
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs.view(outputs.shape[0], label_dim), batch_train_labels)

            #backward pass
            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        print('Train {} for embedding: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}'.format(net_name, epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))

        #save checkpoint
        if path_to_ckpt is not None and (((epoch+1) % 50 == 0) or (epoch+1==epochs)):
            save_file = path_to_ckpt + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(net_name, net_name, epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer_resnet.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net

class label_dataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        super(label_dataset, self).__init__()

        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):

        y = self.labels[index]
        return y

    def __len__(self):
        return self.n_samples

def train_mlp(unique_labels_norm, model_mlp, model_name, model_h2y, epochs=500, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128, device="cuda", label_shape=(1,)):
    '''
    unique_labels_norm: an array of normalized unique labels
    '''

    model_mlp = model_mlp.to(device)
    model_h2y = model_h2y.to(device)

    ''' learning rate decay '''
    def adjust_learning_rate_2(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    if label_shape is None:
        label_shape = (1,)
    elif isinstance(label_shape, (list, tuple)):
        label_shape = tuple(label_shape)
    else:
        label_shape = (label_shape,)
    label_dim = int(np.prod(label_shape)) if label_shape else 1

    reshaped_unique = unique_labels_norm.reshape(-1, *label_shape)

    assert np.max(reshaped_unique)<=1 and np.min(reshaped_unique)>=0
    trainset = label_dataset(reshaped_unique)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model_h2y.eval()
    optimizer_mlp = torch.optim.SGD(model_mlp.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        model_mlp.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_mlp, epoch)
        for _, batch_labels in enumerate(trainloader):

            batch_labels = batch_labels.type(torch.float).to(device)
            batch_size_curr = batch_labels.shape[0]
            batch_labels = batch_labels.reshape(batch_size_curr, label_dim)

            # generate noises which will be added to labels
            batch_gamma = torch.randn_like(batch_labels) * 0.2

            # add noise to labels
            batch_labels_noise = torch.clamp(batch_labels + batch_gamma, 0.0, 1.0)

            #Forward pass
            batch_hiddens_noise = model_mlp(batch_labels_noise)
            batch_rec_labels_noise = model_h2y(batch_hiddens_noise)

            loss = nn.MSELoss()(batch_rec_labels_noise.view(batch_size_curr, label_dim), batch_labels_noise)

            #backward pass
            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        print('\n Train {}: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}'.format(model_name, epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
    #end for epoch

    return model_mlp





# if __name__ == "__main__":
    
#     label_embedding = LabelEmbed(dataset="RC-49", path_y2h="./", path_y2cov="./", type="sinusoidal")
#     y = torch.randn(10, 1).cuda()
#     print(label_embedding.fn_y2h(y).shape)
#     print(label_embedding.fn_y2cov(y).shape)
    
    
    
#     from dataset import LoadDataSet

#     file_path = 'C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/RC-49'  
#     dataset = LoadDataSet(data_name="RC-49", data_path=file_path, min_label=0, max_label=90, img_size=64, max_num_img_per_label=25, num_img_per_label_after_replica=0)
    
#     label_embedding = LabelEmbed(dataset=dataset, path_y2h="./output/model_y2h", path_y2cov="./output/model_y2cov", type="resnet")
    
#     y = torch.randn(10, 1)
    
#     print(label_embedding.fn_y2h(y).shape)
#     print(label_embedding.fn_y2cov(y).shape)
