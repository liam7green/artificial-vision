import os
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import numpy as np
import itertools
import matplotlib.image as mpimg
import argparse
import torch
from tqdm import tqdm
from nn import PretrainedResNetVolumetricContextFPN

def data_generation_stimulus(movie_idx='4', timesteps=20):
    """
    Generates data containing batch_size samples with visual context.
    Returns X of shape (n_samples, timesteps, H, W, C).
    """

    movie_files = [
        '7T_MOVIE1_CC1_v2.mp4',
        '7T_MOVIE2_HO1_v2.mp4',
        '7T_MOVIE3_CC2_v2.mp4',
        '7T_MOVIE4_HO2_v2.mp4'
    ]
    root_data = '../preprocess/Post_20140821_version/'
    videos = [imageio.get_reader(os.path.join(root_data, m), 'ffmpeg')
              for m in movie_files]
    video = videos[int(movie_idx) - 1]
    
    clips = np.load('../preprocess/clip_times_24.npy', allow_pickle=True)
    idxs = clips.item().get(movie_idx)
    
    frame_idx = []
    for c in range(len(idxs) - 1): 
        start_sec, end_sec = idxs[c]

        frame_idx.append(np.arange(start_sec / 24,
                                    end_sec / 24).astype(int))
    frame_idx = np.array(list(itertools.chain(*frame_idx))) * 24
    
    X = []
    pad = timesteps - 1
    for f in frame_idx:

        seq = []
        for t in range(timesteps):
            idx = int(f + t - pad)
            frame = video.get_data(idx)
            seq.append(frame)
        X.append(seq)
    
    return np.asarray(X)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions')
    parser.add_argument('--model_file', default="./models/model_last.pt", type=str, help='Model')
    parser.add_argument('--predictions_file', default="./results", type=str, help='File for saving predictions')
    parser.add_argument('--gpu_device', default="0", type=str, help='GPU device ID')
    parser.add_argument('--max_samples', default=699, type=int, help='Maximum number of samples to process')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for processing')
    parser.add_argument('--valid_corr_file', default="./valid_correlations.npy", type=str, help='File for saving valid correlations')
    args = parser.parse_args()

    def evaluate_corr(Ypred, Yt):
        _Ypred = Ypred[:,:,:,:,0]
        _Ytrue = np.moveaxis(Yt, 3, 0)

        pred_mean = np.mean(_Ypred,0, keepdims=True)
        true_mean = np.mean(_Ytrue,0, keepdims=True)

        pred_std = np.std(_Ypred,0, keepdims=False)
        true_std = np.std(_Ytrue,0, keepdims=False)

        num = np.mean((_Ypred-pred_mean)*(_Ytrue-true_mean),0)
        den = pred_std*true_std

        p = num/den
        return p

    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    model = PretrainedResNetVolumetricContextFPN(
        img_shape=(20, 720, 1024, 3), 
        pretrained=False 
    )
    
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model = model.to(device)
    model.eval() 
    
    X = data_generation_stimulus(movie_idx='4')[:699] 
    
    print('Predicting...')
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), 1):  
            batch = X[i:i+1]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            outputs = model(batch_tensor)
            
            predictions.append(outputs.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(X)} sequences")
    
    Y = np.concatenate(predictions, axis=0)
    
    print('Saving...')
    np.save(args.predictions_file, Y)

if __name__ == '__main__':
    main()