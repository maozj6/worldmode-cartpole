import argparse
# This is a sample Python script.
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mdn_loader import MDNDataset
import argparse

from mdrnn import MDRNN, gmm_loss
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
class VAE(nn.Module):
    def __init__(self, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # Input: (3, 96, 96) -> Output: (32, 48, 48)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (32, 48, 48) -> (64, 24, 24)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (64, 24, 24) -> (128, 12, 12)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (128, 12, 12) -> (256, 6, 6)
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_size)

        # Decoder
        self.dec_fc = nn.Linear(latent_size, 256 * 6 * 6)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.dec_conv5 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2, padding=2)
        # self.dec_conv6 = nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 6, 6)  # Reshape to match the beginning shape of the decoder
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        # x = F.relu(self.dec_conv5(x))
        # x = torch.sigmoid(self.dec_conv6(x))  # Ensure the output is in [0, 1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob


if __name__ == '__main__':
    rnn_dir=''
    SEQ_LEN=32
    BSIZE = 16
    LSIZE = 32
    batchsize = BSIZE

    device="cuda"
    print(device)
    mdrnn = MDRNN(32, 1, 256, 5).to(device)
    parser = argparse.ArgumentParser(description='New latent Trainer')

    parser.add_argument('--train', default="/train/",
                        help='Best model is not reloaded if specified')
    parser.add_argument('--test',default="/test/",
                        help='Does not save samples during training if specified')
    parser.add_argument('--save',default="savedir",
                        help='Does not save samples during training if specified')

    args = parser.parse_args()

    test_path = args.test
    train_path = args.train
    # save_path = args.save

    test_dataset = MDNDataset(test_path)
    test_dataset.load_next_buffer()

    train_dataset = MDNDataset(train_path)
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

    vae = VAE().to(device)
    best = torch.load("real_large_best_model.pth")
    vae.load_state_dict(best)

    vae.eval()


    optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)
    cur_best = None
    train_loss_list=[]
    test_loss_list = []
    for epoch in range(600):
        pbar = tqdm(total=len(train_loader.dataset), desc="Epoch {}".format(epoch))
        cum_train_loss = 0
        mdrnn.train()
        for i, data in enumerate(train_loader):
            inputs, next_obs,acts = data
            inputs = inputs.to(device)
            acts=acts.to(device)
            obs = inputs.float()
            next_obs = next_obs.to(device)
            next_obs = next_obs.float()
            obs=obs.view(-1, 3, 96, 96)
            next_obs=next_obs.view(-1, 3, 96, 96)
            # obs, next_obs = [
            #     f.upsample(x.view(-1, 3, 96, 96), size=96,
            #                mode='bilinear', align_corners=True)
            #     for x in (obs, next_obs)]
            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

            mus, sigmas, logpi, rs, ds = mdrnn(acts.view(BSIZE,32,1), latent_obs)
            gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
            cum_train_loss+=gmm.item()
            # losses = get_loss(latent_obs, action, reward,
            #                   terminal, latent_next_obs, include_reward)
            optimizer.zero_grad()
            gmm.backward()
            optimizer.step()
            pbar.set_postfix_str("loss={loss:10.6f} ".format(
                loss=cum_train_loss / (i + 1),))
            pbar.update(BSIZE)
        pbar.close()

        pbar = tqdm(total=len(test_loader.dataset), desc="Epoch test {}".format(epoch))
        cum_loss=0
        mdrnn.eval()
        for i, data in enumerate(test_loader):
            inputs, next_obs, acts = data
            inputs = inputs.to(device)
            acts = acts.to(device)
            obs = inputs.float()
            next_obs = next_obs.to(device)
            next_obs = next_obs.float()
            obs = obs.view(-1, 3, 96, 96)
            next_obs = next_obs.view(-1, 3, 96, 96)
            # obs, next_obs = [
            #     f.upsample(x.view(-1, 3, 96, 96), size=96,
            #                mode='bilinear', align_corners=True)
            #     for x in (obs, next_obs)]
            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

            mus, sigmas, logpi, rs, ds = mdrnn(acts.view(BSIZE, 32, 1), latent_obs)
            gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
            cum_loss += gmm.item()
            pbar.set_postfix_str("loss={loss:10.6f} ".format(
                loss=cum_loss / (i + 1), ))
            pbar.update(BSIZE)
        pbar.close()
        test_loss = cum_loss
        train_loss_list.append(cum_train_loss)
        test_loss_list.append(cum_loss)
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        checkpoint_fname =  'realmdncheckpoint'+'.tar'
        save_checkpoint({
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "precision": test_loss,
            "train_loss":train_loss_list,
            "test_loss":test_loss_list,
            "epoch": epoch}, is_best, checkpoint_fname,
            'realmdnbest'+'.tar')

        # print("end")
