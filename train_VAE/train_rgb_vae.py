import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rgb_loader import VaeDataset
from tqdm import tqdm
import torch.nn.functional as F
# Define the VAE architecture
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence



if __name__ == '__main__':
    device = "cuda"
    print(device)
    print(torch.cuda.is_available())
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train', type=str,default="/home/mao/24Spring/Clip/cartpoledata/496big/train/", help='Path to input file')
    parser.add_argument('--test', type=str,default="/home/mao/24Spring/Clip/cartpoledata/496big/test/", help='Path to input file')
    parser.add_argument('--save', type=str, default='/home/mao/24Spring/Clip/cartpoledata/4statedata/train/controller1/',
                        help='Path to output dir ')

    args = parser.parse_args()

    trainpath=args.train
    # Set up the training parameters
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 2000
    best_loss = float('inf')
    # Initialize the VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # # Load your dataset
    # transform = transforms.Compose([
    #     transforms.Resize((96, 96)),
    #     transforms.ToTensor()
    # ])
    #
    traindataset = VaeDataset(root=trainpath)
    TrainLoader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    testdataset = VaeDataset(root=args.test)
    TestLoader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    # Train the VAE
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(TrainLoader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Training epoch " + str(epoch))
        for batch_idx, data in enumerate(TrainLoader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloader.dataset),
            #                100. * batch_idx / len(dataloader), loss.item() / len(data)))

            pbar.update(1)
        pbar.close()

        print('  ====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / (len(TrainLoader)*batch_size)))
        model.eval()
        pbar = tqdm(total=len(TestLoader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Test epoch " + str(epoch))
        total_loss = 0

        for batch_idx, data in enumerate(TestLoader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloader.dataset),
            #                100. * batch_idx / len(dataloader), loss.item() / len(data)))

            pbar.update(1)
        save_image(data, 'figs_vae/gt_{}.png'.format(epoch), range=(0, 1))
        save_image(recon_batch, 'figs_vae/reconstructed_{}.png'.format(epoch), range=(0, 1))
        pbar.close()
        print('  ====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / (len(TestLoader)*batch_size)))
        testloss=total_loss / (len(TestLoader)*batch_size)
        scheduler.step(testloss)
        if testloss < best_loss:
            best_loss = testloss
            # 保存最佳模型的参数
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved Best Model with Loss: {best_loss:.4f}")

            # 同时保存检查点
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'check_point_model.pth')
            print(f"Checkpoint saved at 'check_point_model.pth'")
    # Save the trained model
