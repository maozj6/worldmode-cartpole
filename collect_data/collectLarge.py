import cv2
from datetime import datetime
# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
import gym
import tkinter
USE_CUDA = True # If we want to use GPU (powerful one needed!)
env = gym.make('CartPole-v0').unwrapped

GRAYSCALE = True # False is RGB
RESIZE_PIXELS = 60 # Downsample image to this number of pixels
FRAMES = 2 # state is the number of last frames: the more frames,
device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

if GRAYSCALE == 0:
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.ToTensor()])

    nn_inputs = 3 * FRAMES  # number of channels for the nn
else:
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.Grayscale(),
                        T.ToTensor()])
    nn_inputs = FRAMES  # number of channels for the nn

# ---- CONVOLUTIONAL NEURAL NETWORK ----
HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 32
KERNEL_SIZE = 5 # original = 5
STRIDE = 2 # original = 2
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

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

n_actions = env.action_space.n

def show(image):
    plt.imshow(image)  # Transpose to (400, 600, 3) for displaying
    plt.axis('on')  # Turn off axis
    plt.show()
# def get_screen():
#     # Returned screen requested by gym is 400x600x3, but is sometimes larger
#     # such as 800x1200x3. Transpose it into torch order (CHW).
#     # screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#
#     img  = env.render(mode='rgb_array')
#
#     new_image_array = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 255 represents white
#
#     # Copy the content of the original image to the new image
#     new_image_array[100:500, :, :] = img
#     # image_pil.save("obs2.png")  # Transpose to (400, 600, 3)
#     # resized_image_array = np.resize(new_image_array, (96, 96, 3))
#     # image_pil2 = Image.fromarray(resized_image_array)
#     # image_pil2.save("obs3.png")  # Transpose to (400, 600, 3)
#     resized_image_array = cv2.resize(new_image_array, (96, 96))
#     resized_image_array[63:64, :, :] = [0, 0, 0]  # Set pixel values to zero (black)
#     # image_pil = Image.fromarray(resized_image_array)
#     screen = resized_image_array
#     resized_screen = cv2.resize(screen, (135, 135))
#     screen = resized_screen[35:95]
#     # show(screen)
#     gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#     # cv2.imwrite("./obs.png", screen)
#     # image_pil = Image.fromarray(screen.transpose(1, 2, 0))
#     # image_pil.save("obs.png")# Transpose to (400, 600, 3)
#     # Cart is in the lower half, so strip off the top and bottom of the screen
#     # _, screen_height, screen_width = screen.shape
#     # screen=resized_screen[100:500].transpose((2, 0, 1))
#     # # show(resized_screen)
#     # _, screen_height, screen_width = screen.shape
#     # screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
#     # view_width = int(screen_width * 0.6)
#     # cart_location = get_cart_location(screen_width)
#     # if cart_location < view_width // 2:
#     #     slice_range = slice(view_width)
#     # elif cart_location > (screen_width - view_width // 2):
#     #     slice_range = slice(-view_width, None)
#     # else:
#     #     slice_range = slice(cart_location - view_width // 2,
#     #                         cart_location + view_width // 2)
#     # # Strip off the edges, so that we have a square image centered on a cart
#     # screen = screen[:, :, slice_range]
#     # # Convert to float, rescale, convert to torch tensor
#     # # (this doesn't require a copy)
#     # show(screen)
#     screen = np.ascontiguousarray(gray_image, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     final = resize(screen).unsqueeze(0).to(device)
#     # Resize, and add a batch dimension (BCHW)
#     return final


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    img = env.render(mode='rgb_array')

    new_image_array = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 255 represents white

    new_image_array[100:500, :, :] = img

    resized_image_array = cv2.resize(new_image_array, (96, 96))
    resized_image_array[63:64, :, :] = [0, 0, 0]  # Set pixel values to zero (black)
    # image_pil = Image.fromarray(resized_image_array)
    screen = resized_image_array
    resized_screen = cv2.resize(screen, (135, 135))
    screen = resized_screen[35:95]
    gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    screen = np.ascontiguousarray(gray_image, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    final = resize(screen).unsqueeze(0).to(device)
    return final

env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
stop_training = False

if __name__ == '__main__':

    seedlist = [0,0,0,123,456,789]
    totallist=[15000, 15000, 15000, 45000, 45000, 45000]
    modelpathlist = ["newmodels/new1.pth","newmodels/new2.pt","newmodels/new3.pth",
                     "newmodels/new1.pth","newmodels/new2.pt","newmodels/new3.pth"]
    savelist= ['/home/mao/24Spring/Clip/cartpoledata/496_real_large/test/controller1/',
               '/home/mao/24Spring/Clip/cartpoledata/496_real_large/test/controller2/',
               '/home/mao/24Spring/Clip/cartpoledata/496_real_large/test/controller3/',
               '/home/mao/24Spring/Clip/cartpoledata/496_real_large/train/controller1/',
               '/home/mao/24Spring/Clip/cartpoledata/496_real_large/train/controller2/',
               '/home/mao/24Spring/Clip/cartpoledata/496_real_large/train/controller3/']
    for collectNumber in range(6):
        env.seed(seedlist[collectNumber])
        np.random.seed(seedlist[collectNumber])
        total = totallist[collectNumber]
        theta_thre=45 * 2 * math.pi / 360
        model1 = torch.load(modelpathlist[collectNumber])



        model = DQN(screen_height, screen_width, n_actions).to(device)
        model.load_state_dict(model1)
        collect = 0
        unsafe_collect=0
        npz_guard = 0
        outdir = savelist[collectNumber]
        pbar = tqdm(total=total,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Collecting files " + str(npz_guard))

        while collect < total:
            env.reset()
            guard = 0

            init_screen = get_screen()
            screens = deque([init_screen] * FRAMES, FRAMES)
            END = False

            recording_safe = []
            recording_action = []

            recording_obs = []
            recording_label=[]
            recording_rad=[]
            recording_vel=[]
            recording_angvel=[]
            recording_x96=[]

            recording_degree = []
            recording_position = []
            numberList = []
            while not END:

                state = torch.cat(list(screens), dim=1)

                action=model(state).max(1)[1].view(1, 1)
                state_variables, _, done, (theta2,x2) = env.step(action.item())
                env.render()
                position = state_variables[0]
                velocity = state_variables[1]
                radangle = state_variables[2]
                anglevelocity = state_variables[3]

                END= done
                #right+ left- angles and both
                degree = radangle*180/math.pi
                x96 = position*20
                if END:
                    print("end")
                screens.append(get_screen())
                recording_action.append(action.cpu().detach().numpy().item())
                # print(recording_action)
                img = screen = env.render(mode='rgb_array')
                new_image_array = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 255 represents white

                # Copy the content of the original image to the new image
                new_image_array[100:500, :, :] = img
                image_pil = Image.fromarray(new_image_array)
                image_pil.save("obs2.png")# Transpose to (400, 600, 3)
                # resized_image_array = np.resize(new_image_array, (96, 96, 3))
                # image_pil2 = Image.fromarray(resized_image_array)
                # image_pil2.save("obs3.png")  # Transpose to (400, 600, 3)
                resized_image_array = cv2.resize(new_image_array, (96, 96))
                resized_image_array[63:64, :, :] = [0, 0, 0]  # Set pixel values to zero (black)
                # image_pil2 = Image.fromarray(resized_image_array)
                # image_pil2.save("obs3.png")  # Transpose to (400, 600, 3)
                # transposed = resized_image_array.transpose((2, 0, 1))
                # fill = np.ones((30,135),float)
                # fill2 = np.ones((45,135),float)
                # img=np.concatenate((fill,img ), axis=0)
                #
                # img=np.concatenate((img, fill2), axis=0)
                # obs = cv2.resize(img, (96, 96))
                # cv2.imwrite("obs/"+str(guard)+".png",obs*255)
                recording_obs.append(resized_image_array)
                recording_degree.append(degree)
                recording_rad.append(radangle)
                recording_vel.append(radangle)
                recording_angvel.append(radangle)
                recording_x96.append(x96)
                recording_position.append(position)

                next_state = torch.cat(list(screens), dim=1)
                state = next_state
                if (radangle < -theta_thre or radangle > theta_thre):
                    safe = 0
                    unsafe_collect=unsafe_collect+1
                else:
                    safe= 1
                # theta=(theta*360)/(2 * math.pi )
                # print(theta)
                # print()
                # print(safe)
                recording_safe.append(safe)
                guard = guard+1
                # print(guard)

            if guard>15:
                for big_i in range(len(recording_safe) - 10):
                    labels = []
                    initial_safe = True
                    small_i = 0
                    while len(labels) < 10:
                        # print(big_i)
                        # print("b and s")
                        # print(small_i)
                        if recording_safe[big_i + small_i] == 0:
                            for little_j in range(10 - len(labels)):
                                labels.append(0)
                            break
                        labels.append(recording_safe[big_i + small_i])
                        small_i = small_i + 1
                    recording_label.append(labels)
                current_date_time = datetime.now()

                # Format the date as YYMMDDHHMM
                formatted_date_time = current_date_time.strftime('%y%m%d%H%M%S')
                numberList.append(len(recording_obs))
                print("mean obs: "+str(np.mean(numberList)))
                np.savez_compressed(outdir + "/" + str(formatted_date_time) + ".npz", obs=recording_obs,
                                    action=recording_action, safe=recording_safe,label=recording_label,x=recording_position,
                                    degree=recording_degree,rad=recording_rad,x96=recording_x96,vel=recording_vel,angvel=recording_angvel
                                   )



                npz_guard =npz_guard+1
                collect=collect+guard
                print(collect)
                print(unsafe_collect/collect)
                pbar.update(guard)

        pbar.close()


