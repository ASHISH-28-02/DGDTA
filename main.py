import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import sys

# --- FIX: Changed import to use GATv2Model ---
from models.gatv2 import GATv2Model
from utils import TestbedDataset, rmse, mse, pearson, spearman, ci


def train_epoch(model, device, data_loader, optimizer, epoch, log_interval):
    """Function to train the model for one epoch."""
    print(f'Training on {len(data_loader.dataset)} samples...')
    model.train()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(data_loader.dataset)} '
                  f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate_model(model, device, data_loader):
    """Function to evaluate the model on a given dataset."""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Making prediction for {len(data_loader.dataset)} samples...')
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    """Main function to run the training and evaluation pipeline."""
    datasets = ['kiba', 'davis']
    
    # --- FIX: Set the model to GATv2Model ---
    model_class = GATv2Model
    model_name = model_class.__name__

    cuda_device = "cuda:0"
    
    # Using a smaller, safer batch size
    train_batch_size = 64
    test_batch_size = 64
    learning_rate = 0.0005
    log_interval = 20
    num_epochs = 1000
    patience = 50

    print('Learning rate:', learning_rate)
    print('Epochs:', num_epochs)
    print(f'Train Batch Size: {train_batch_size}')
    print(f'Early Stopping Patience: {patience}')


    for dataset in datasets:
        print(f'\nRunning on {model_name}_{dataset}')
        train_file = f'data/processed/{dataset}_train.pt'
        test_file = f'data/processed/{dataset}_test.pt'
        if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
            print('Please run create_data.py to prepare data in PyTorch format!')
            continue

        full_train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')

        train_size = int(0.8 * len(full_train_data))
        valid_size = len(full_train_data) - train_size
        train_data, valid_data = random_split(full_train_data, [train_size, valid_size])

        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

        device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_valid_mse = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        
        model_file = f'model_{model_name}_{dataset}.model'
        result_file = f'result_{model_name}_{dataset}.csv'
        
        # --- FIX: Initialize the DataFrame before the loop ---
        results_history = []

        for epoch in range(1, num_epochs + 1):
            train_epoch(model, device, train_loader, optimizer, epoch, log_interval)
            
            print('Predicting for validation data...')
            valid_labels, valid_preds = evaluate_model(model, device, valid_loader)
            valid_mse = mse(valid_labels, valid_preds)

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_epoch = epoch
                torch.save(model.state_dict(), model_file)
                epochs_no_improve = 0
                
                print('Validation MSE improved. Evaluating on test set...')
                test_labels, test_preds = evaluate_model(model, device, test_loader)
                test_metrics = [rmse(test_labels, test_preds), mse(test_labels, test_preds), 
                                pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds)]
                
                with open(result_file, 'w') as f:
                    f.write('rmse,mse,pearson,spearman,ci\n')
                    f.write(','.join(map(str, test_metrics)))
                
                print(f'Epoch {best_epoch} - Best Model Saved. Test MSE: {test_metrics[1]:.4f}, Test CI: {test_metrics[4]:.4f}')
                
                epoch_results = {'epoch': epoch, 'valid_mse': valid_mse}
                epoch_results.update({
                    'test_rmse': test_metrics[0], 'test_mse': test_metrics[1],
                    'test_pearson': test_metrics[2], 'test_spearman': test_metrics[3], 'test_ci': test_metrics[4]
                })
                results_history.append(epoch_results)

            else:
                epochs_no_improve += 1
                print(f'Validation MSE did not improve for {epochs_no_improve} epochs. Best at epoch {best_epoch}: {best_valid_mse:.4f}')
            
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {patience} epochs with no improvement.')
                break
        
        history_df = pd.DataFrame(results_history)
        history_df.to_csv(f'full_results_history_{model_name}_{dataset}.csv', index=False)
        print(f"\nTraining finished. Full results history saved to full_results_history_{model_name}_{dataset}.csv")

if __name__ == "__main__":
    main()
