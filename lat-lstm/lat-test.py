
from os import listdir
from os.path import join, isdir
import numpy as np
from tqdm import tqdm
import argparse
from tqdm import tqdm
from torch.distributions.categorical import Categorical
import cv2
from collections import deque

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image

HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
nn_inputs=2

class latent_lstm(nn.Module):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents=32, actions=1, hiddens=256):
        super().__init__()

        self.rnn = nn.LSTMCell(latents + actions, hiddens)
        self.fc = nn.Linear(
            hiddens, 32)

    def forward(self, actions, latents,hidden): # pylint: disable=arguments-differ
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins,hidden)
        out=self.fc(outs)
        return out
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(nn_inputs, HIDDEN_LAYER_1, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn1 = nn.BatchNorm2d(HIDDEN_LAYER_1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER_1, HIDDEN_LAYER_2, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(HIDDEN_LAYER_2)
        self.conv3 = nn.Conv2d(HIDDEN_LAYER_2, HIDDEN_LAYER_3, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(HIDDEN_LAYER_3)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        nn.Dropout()
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
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
RESIZE_PIXELS = 60 # Downsample image to this number of pixels

resize = T.Compose([T.ToPILImage(),
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.ToTensor()])
def adapt2controller(screen):
    screen = (screen * 255).astype(np.uint8)
    resized_screen = cv2.resize(screen, (135, 135))
    screen = resized_screen[35:95]
    gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    screen = np.ascontiguousarray(gray_image, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    final = resize(screen).unsqueeze(0).to(device)
    return final
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-r', '--root', type=str,default="/home/mao/24Spring/Clip/cartpoledata/496_real_large/test/controller1/", help='the path of saved model')
    parser.add_argument('-n', '--name', type=str,default="c1rsuper", help='the path of saved results')
    parser.add_argument('-m', '--mdn', type=str,default="reallatlstmcheckpoint.tar", help='the path of mdn model')
    parser.add_argument('-e', '--eva', type=str,default="evabest.pth", help='the path of eva model')
    parser.add_argument('-v', '--vae', type=str,default="large_best_model.pth", help='the path of vae model')
    parser.add_argument('-c', '--ctrl', type=str,default="./newmodels/new2.pt", help='the path of vae model')
    device="cuda"
    print(device)
    # parser.add_argument('-d', '--dir', default='./data/test/ctrl2/', help='output path, dir\'s name')
    # parser.add_argument('-c', '--controller', default='save/trial_200.onnx',
    #                     help='The number of episodes should the model plays.')
    args = parser.parse_args()
    root = args.root
    model1 = torch.load(args.ctrl)

    model = DQN(60, 135, 2).to(device)
    model.load_state_dict(model1)

    # import wandb
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="cart-fwm"+args.name,
    #     # track hyperparameters and run metadata
    #     config={
    #     }
    # )
    inumberlist = [8,4,2,1,16,32]
    prednumberlist = [30,20,30,40,50,60]

    mdrnn = latent_lstm(32, 1, 256).to(device)
    rnn_state = torch.load(args.mdn, map_location=lambda storage, location: storage)
    rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
    mdrnn.load_state_dict(rnn_state_dict)
    LSIZE = 32
    FRAMES = 2  # state is the number of last frames: the more frames,

    vae = VAE().to(device)
    best = torch.load(args.vae)
    vae.load_state_dict(best)
    MSE = nn.MSELoss()

    vae.eval()
    mdrnn.eval()

    evaluator = CNN().to(device)
    evaluator.load_state_dict(torch.load(args.eva))
    evaluator.eval()
    for largei in range(0,len(inumberlist)):
        pbar = tqdm(total=len(prednumberlist),
                    )
        for largej in range(0,len(prednumberlist)):
            pbar.update(1)

            inumber = inumberlist[largei]
            prednumber = prednumberlist[largej]
            mselist = np.zeros((prednumber))
            ssimlist = np.zeros((prednumber))
            tempgtlist = []
            temppredlist = []
            tempgtlist5 = []
            temppredlist5 = []
            xlist = []
            anglelist = []
            tempsafelist = []
            evaluatorsafety = []
            tempactionlist = []
            files = []
            for sd in listdir(root):
                if isdir(join(root, sd)):
                    for ssd in listdir(root + sd):
                        files.append(join(root, sd, ssd))
                else:
                    files.append(join(root, sd))
            data = np.load(files[0])
            files.sort()
            sequencedegree = []
            sequencex = []
            sequenceaction = []
            sequencesafe = []
            sequenceobs = []

            for i in range(len(files)):

                degree = data['degree']
                x96 = data['x96']
                action = data['action']
                obs = data['obs']
                safe = data['safe']

                frontnumber =len(degree) -50-inumber-prednumber

                for j in range(50):

                    degreesmalllist = []
                    xsmalllist = []
                    actionsmall = []
                    safesmall = []
                    obssmall = []
                    basenumber = frontnumber + j
                    for k in range(inumber + prednumber):
                        degreesmalllist.append(degree[basenumber + k])
                        xsmalllist.append(x96[basenumber + k])
                        actionsmall.append(action[basenumber + k])
                        obssmall.append(obs[basenumber + k])
                        safesmall.append(safe[basenumber + k])

                    sequencedegree.append(degreesmalllist)
                    sequencex.append(xsmalllist)
                    sequenceaction.append(actionsmall)
                    sequenceobs.append(obssmall)
                    sequencesafe.append(safesmall)

            # data = [1, 3, 5, 7, 9, 11]
            # np.savez_compressed(str(inumber)+"inp"+str(prednumber)+".npz",degree=sequencedegree,x96=sequencex,action=sequenceaction,obs=sequenceobs,safe=sequencesafe)
            for i in range(len(sequenceobs)):
                inputs = np.array(sequenceobs[i][0:inumber]).transpose((0, 3, 1, 2))
                next_obs = np.array(sequenceobs[i][inumber:inumber + prednumber]).transpose((0, 3, 1, 2))

                inputs = np.ascontiguousarray(inputs, dtype=np.float32) / 255
                inputs = torch.from_numpy(inputs).to(device)
                acts = sequenceaction[i]
                acts = torch.tensor(acts).float().to(device)

                next_obs = np.ascontiguousarray(next_obs, dtype=np.float32) / 255
                next_obs = torch.from_numpy(next_obs).to(device)

                (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                    vae(x)[1:] for x in (inputs, next_obs)]

                latent_obs = [
                    (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(1, len(inputs), LSIZE)
                    for x_mu, x_logsigma in
                    [(obs_mu, obs_logsigma)]]

                latent_next_obs = [
                    (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(1, len(next_obs), LSIZE)
                    for x_mu, x_logsigma in
                    [(next_obs_mu, next_obs_logsigma)]]
                hidden = 2 * [torch.zeros(1, 256).to(device)]
                for initi in range(len(inputs)):
                    zoutput = mdrnn(acts[initi].view(1, 1),latent_obs[0][0][initi].view(1, 32), hidden)

                # mixt = Categorical(torch.exp(logpi.squeeze())).sample().item()
                # latent_state = mus[:, mixt, :]  # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])

                decoded_obs = vae.decode(zoutput).squeeze(0)
                np_obs = decoded_obs.cpu().detach().numpy().transpose((1, 2, 0))
                savelist = []
                # for tempi in range(len(inputs)):
                #     image = inputs[tempi]
                #     save_image(image, 'input_{}.png'.format(tempi), range=(0, 1))
                shapedobs = adapt2controller(np_obs)
                screens = deque([shapedobs] * FRAMES, FRAMES)

                for initi in range(len(next_obs)):
                    thistempaction = decoded_obs
                    state = torch.cat(list(screens), dim=1)

                    thisacts = model(state).max(1)[1].view(1, 1)
                    # mus, sigmas, logpi, rs, ds, hidden = mdrnn(acts[0][initi].view(1,1), latent_obs[0][0][initi].view(1,32), hidden)
                    zoutput = mdrnn(acts[initi].view(1, 1),latent_obs[0][0][initi].view(1, 32), hidden)
                    # mixt = Categorical(torch.exp(logpi.squeeze())).sample().item()
                    # latent_state = mus[:, mixt, :]  # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
                    decoded_obs = vae.decode(zoutput)

                    np_obs = decoded_obs.squeeze(0).cpu().detach().numpy().transpose((1, 2, 0))

                    screens.append(adapt2controller(np_obs))

                    predsafetysinle = evaluator(decoded_obs)
                    _, predictedsafeyt = torch.max(predsafetysinle, 1)
                    evaluatorsafety.append(predictedsafeyt.item())
                    np_obs = decoded_obs.cpu().detach().numpy()
                    reconstruction_loss = MSE(next_obs[initi].view(3,96,96), decoded_obs.view(3,96,96))
                    # predlist[initi].append(np_obs)
                    mselist[initi] += reconstruction_loss.item()
                    arr_reshaped = np_obs.squeeze().transpose(1, 2, 0)
                    # Scale the values from the range (0-1) to (0-255)
                    arr_scaled = arr_reshaped * 255
                    # Convert the array to integers
                    arr_scaled_int = arr_scaled.astype(np.uint8)
                    gray_image1 = cv2.cvtColor(arr_scaled_int, cv2.COLOR_BGR2GRAY)
                    save_image(next_obs[initi], './saveimage/input_{}.png'.format(initi), range=(0, 1))

                    groundtruthimage = next_obs[initi].cpu().detach().numpy()
                    groundtruthimage=groundtruthimage.squeeze().transpose(1, 2, 0)
                    groundtruthimage = groundtruthimage * 255
                    # Convert the array to integers
                    groundtruthimage = groundtruthimage.astype(np.uint8)
                    if initi==len(next_obs)-1:
                        tempgtlist.append(groundtruthimage)
                        temppredlist.append(arr_scaled_int)
                    if initi == len(next_obs) - 6:
                        tempgtlist5.append(groundtruthimage)
                        temppredlist5.append(arr_scaled_int)
                    gray_image2 = cv2.cvtColor(groundtruthimage, cv2.COLOR_BGR2GRAY)
                    ssim_index = ssim(gray_image1, gray_image2)
                    ssimlist[initi]+=ssim_index
                inputs = np.array(sequenceobs[i][0:inumber]).transpose((0, 3, 1, 2))
                next_obs = np.array(sequenceobs[i][inumber:inumber + prednumber]).transpose((0, 3, 1, 2))
                xlist.append(sequencex[i][0:inumber + prednumber])
                anglelist.append(sequencedegree[i][0:inumber + prednumber])
                tempsafelist.append(sequencesafe[i][0:inumber + prednumber])
                tempactionlist.append(sequenceaction[i][0:inumber + prednumber])
            print( mselist/ len(sequenceobs))
            print( ssimlist/ len(sequenceobs))

            plt.plot(mselist/ len(sequenceobs))
            plt.xlabel('steps')
            plt.ylabel('MSE')
            plt.title('MSE Plot for'+str(inumber)+' and '+str(prednumber))
            directory = './'+args.name+'/'


            plt.savefig(directory+str(inumber)+' mse '+str(prednumber)+'.png')

            plt.clf()


            plt.plot(ssimlist/ len(sequenceobs))
            plt.xlabel('horizon')
            plt.ylabel('ssim')
            plt.title('ssim for '+str(inumber)+' and '+str(prednumber))
            plt.savefig(directory+str(inumber)+' ssim '+str(prednumber)+'.png')
            plt.clf()
            np.savez_compressed(directory+str(inumber)+' mse '+str(prednumber)+'.npz',action = tempactionlist,safe = tempsafelist,degree=anglelist,x96 = xlist,predsafe = evaluatorsafety,gt5=tempgtlist5,pred5=temppredlist5,gtobs =tempgtlist,predobs = temppredlist, mse =mselist/ len(sequenceobs),ssim =ssimlist/ len(sequenceobs)  )


        pbar.close()
    print("end")
