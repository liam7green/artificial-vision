import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio

class DataGeneratorVolContext(Dataset):
    'Generates data for PyTorch'
    def __init__(self, 
                 list_IDs,
                 train=True,
                 split=[0.9, 0.1],
                 batch_size=16,
                 fraction=0,
                 dim=(720, 1024),
                 volume=(111, 127, 111),
                 n_channels=3,
                 time_steps=20,
                 delay=None,
                 shuffle=True,
                 stimulus_dir='/Volumes/HCPDataset/Stimulus/Post_20140821_version/',
                 response_dir='/Volumes/HCPDataset/ResponseData/'):
        'Initialization'
        
        self.movie_files = [
            '7T_MOVIE1_CC1_v2.mp4', 
            '7T_MOVIE2_HO1_v2.mp4', 
            '7T_MOVIE3_CC2_v2.mp4', 
            '7T_MOVIE4_HO2_v2.mp4'
        ]

        self.fraction = fraction
        self.root_data = stimulus_dir
        self.vol_root = response_dir
        
        self.delay = delay
        self.time_steps = time_steps
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path, 'ffmpeg'))
        
        self.dim = dim
        self.volume = volume
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'
        X = np.zeros((self.batch_size, self.time_steps, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.volume, 1))

        for i, idx in enumerate(list_indexes):
            subject, movie, frame = self.list_IDs[idx]
            for t in range(self.time_steps):
                frame_index = int(frame) - 12 * self.fraction + 24 * (t - (self.time_steps - 1))
                X[i, t] = np.array(self.videos[int(movie) - 1].get_data(frame_index))
            y[i, :, :, :, 0] = np.load(os.path.join(self.vol_root, subject, f'MOVIE{movie}_MNI.npy'), mmap_mode='r')[2:, 4:-5, :-2, int(int(frame) / 24) + self.delay]

        return X, y