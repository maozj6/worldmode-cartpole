import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evaloader import VaeDataset
from tqdm import tqdm
import torch.nn.functional as F
# Define the VAE architecture
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the input for the fully connected layers
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    device = "cuda"
    print(device)
    print(torch.cuda.is_available())
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train', type=str,default="/home/mao/24Spring/Clip/cartpoledata/496_real_large/train/", help='Path to input file')
    parser.add_argument('--test', type=str,default="/home/mao/24Spring/Clip/cartpoledata/496_real_large/test/", help='Path to input file')
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
    model = CNN().to(device)

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
    criterion = nn.CrossEntropyLoss()
    # Train the VAE
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(TrainLoader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Training epoch " + str(epoch))
        whole = 0
        totalcorrect = 0
        for batch_idx, data in enumerate(TrainLoader):
            data,labels = data
            data = data.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            whole+=batch_size
            totalcorrect+=correct
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloader.dataset),
            #                100. * batch_idx / len(dataloader), loss.item() / len(data)))

            pbar.update(1)
        pbar.close()
        print('Accuracy: {:.2f}%'.format(totalcorrect/whole * 100))
        print('  ====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / (len(TrainLoader)*batch_size)))
        model.eval()
        whole = 0
        totalcorrect = 0
        pbar = tqdm(total=len(TestLoader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Test epoch " + str(epoch))
        total_loss = 0

        for batch_idx, data in enumerate(TestLoader):
            data, labels = data
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            whole += batch_size
            totalcorrect += correct
            # if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloader.dataset),
            #                100. * batch_idx / len(dataloader), loss.item() / len(data)))

            pbar.update(1)
        pbar.close()
        print('  ====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / (len(TestLoader)*batch_size)))
        testloss=total_loss / (len(TestLoader)*batch_size)
        scheduler.step(testloss)
        print('Accuracy: {:.2f}%'.format(totalcorrect/whole * 100))

        if testloss < best_loss:
            best_loss = testloss
            # 保存最佳模型的参数
            torch.save(model.state_dict(), './real_large_best_model.pth')
            print(f"Saved Best Model with Loss: {best_loss:.4f}")

            # 同时保存检查点
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'real_large_check_point_model.pth')
            print(f"Checkpoint saved at 'check_point_model.pth'")
    # Save the trained model
