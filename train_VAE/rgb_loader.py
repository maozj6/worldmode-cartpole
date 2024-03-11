""" Some data loading utilities """
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, buffer_size=0, leng=0): # pylint: disable=too-many-arguments
        self.leng=leng
        self._files=[]
        self.safeCache=0
        for sd in listdir(root):
            if isdir(join(root, sd)):
                for ssd in listdir(root+sd):
                    self._files.append(join(root,sd,ssd))
            else:
                self._files.append(join(root, sd))

        self._files.sort()
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        # self._buffer_size = buffer_size
        self._buffer_size = len(self._files)

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                tmp = {}
                # tmp['actions'] =data['action']
                tmp['observations'] =data['obs']
                tmp['labels'] =data['label']
                tmp['actions'] =data['action']
                tmp['safes'] =data['safe']
                tmp['x'] =data['x']
                tmp['degree'] =data['degree']

                self._buffer.append(tmp)
                # for k, v in data.items():
                #
                #     self._buffer.append({k: np.copy(v)})

                #     self._buffer +=[k: np.copy(v)]
                # self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['safe'].shape[0]-self.leng)]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        while True:
            number=np.random.randint(0,self._cum_size[-1])

            i=number
            file_index = bisect(self._cum_size, i) - 1
            seq_index = i - self._cum_size[file_index]
            data = self._buffer[file_index]
            safes = data['safes'][seq_index +self.leng]
            if safes!=self.safeCache:
                self.safeCache=safes
                # print(safes)
                break

        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass



class VaeDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of images

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args test: if True, test data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def change_background_to_grey(self,obs):
        for i in range(96):
            for j in range(96):
                if obs[i][j] < 1:
                    obs[i][j] = 0.1
        for i in range(96):
            for j in range(96):
                if obs[i][j] == 1:
                    obs[i][j] = 0.75
        ot = gaussian_filter(obs, sigma=2)
        # img = ot
        # Convert numpy array back to PIL Image
        # new_image = Image.fromarray(np_image)
        return ot
    def _get_data(self, data, seq_index):
        img = data['observations'][seq_index]
        # new_image = Image.fromarray(img)
        # grayscale_img = new_image.convert('L')
        # img = np.array(grayscale_img)
        obs = torch.tensor(img)
        tensor_img = obs.permute(2, 0, 1)
        #
        # # Normalize the tensor to the range [0, 1]
        tensor_img = tensor_img.float() / 255.0
        # tensor_img = self.change_background_to_grey(tensor_img)

        return tensor_img


