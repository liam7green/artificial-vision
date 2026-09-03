import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from nn import PretrainedResNetVolumetricContextFPN
from data import DataGeneratorVolContext
from losses import LossHistory, ValLossHistory
from tqdm import tqdm
import time
import boto3
import json
from datetime import datetime
# from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser(description='Contextual model')
    parser.add_argument('--lrate', default=0.0008, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lastckpt_file', default='model_last.pt', help='Location for saving last model')
    parser.add_argument('--model_file', default='model', help='Location for saving model')
    parser.add_argument('--log_file', default='training.log', help='Location for saving logs')
    parser.add_argument('--gpu_devices', default="0", type=str, help='Device IDs')
    parser.add_argument('--pretrained', default=1, type=int, help='Use pretrained ResNet weights')
    parser.add_argument('--delay', default=4, type=int, help='HR')
    parser.add_argument('--timesteps', default=20, type=int, help='Timesteps')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs('results', exist_ok=True)
    
    log_path = os.path.join('results', args.log_file)
    training_log = {
        'args': vars(args),
        'epochs': [],
        'best_val_loss': float('inf'),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_devices}")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    IDs_train = np.genfromtxt('./preprocesscopy/ListIDs_train.txt', dtype='str')
    IDs_val = np.genfromtxt('./preprocesscopy/ListIDs_val.txt', dtype='str')
    
    train_dataset = DataGeneratorVolContext(
        IDs_train, 
        dim=(360, 512), 
        train=True, 
        delay=args.delay, 
        time_steps=args.timesteps,
        batch_size=args.batch_size
    )
    
    val_dataset = DataGeneratorVolContext(
        IDs_val, 
        dim=(360, 512), 
        train=False, 
        delay=args.delay, 
        time_steps=args.timesteps,
        batch_size=args.batch_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=None,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=None,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        )
    
    model = PretrainedResNetVolumetricContextFPN(
        img_shape=(args.timesteps, 360, 512, 3),
        pretrained=bool(args.pretrained)
    )
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lrate)
    criterion = torch.nn.MSELoss()
    
    train_history = LossHistory()
    val_history = ValLossHistory()
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # Create progress bar for training
        train_pbar = tqdm(enumerate(train_loader), 
                         total=min(len(train_loader), 2000),
                         desc=f'Epoch {epoch+1}/{args.epochs}',
                         leave=True)
        
        for batch_idx, (data, target) in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.6f}'
            })
            
            if batch_idx >= 2000:
                break
        
        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(enumerate(val_loader), 
                       total=min(len(val_loader), 100),
                       desc='Validation',
                       leave=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in val_pbar:
                if batch_idx >= 100:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                batch_loss = criterion(output, target).item()
                val_loss += batch_loss
                val_pbar.set_postfix({'val_loss': f'{batch_loss:.6f}'})
        
        val_loss /= min(len(val_loader), 100)
        val_history.on_batch_end(torch.tensor(val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join("models", f"{args.model_file}-{epoch:02d}.pt")
            torch.save(model.state_dict(), model_path)
            print(f'Saved model with best validation loss: {val_loss:.6f}')
        
        epoch_time = time.time() - start_time
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss/min(len(train_loader), 2000),
            'val_loss': val_loss,
            'time': epoch_time,
            'best_val_loss': best_val_loss if val_loss == best_val_loss else None
        }
        training_log['epochs'].append(epoch_metrics)
        
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=4)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Training Loss: {epoch_loss/min(len(train_loader), 2000):.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        print(f'Time: {epoch_time:.2f}s')
    
    final_model_path = os.path.join("models", args.lastckpt_file)
    torch.save(model.state_dict(), final_model_path)
    
    training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    
    print(f"\nTraining log saved to: {log_path}")
    print("\nTraining completed!")

if __name__ == '__main__':
    main()