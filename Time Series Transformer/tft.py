import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
def load_data(data_string):
    # Convert string to DataFrame
    data = pd.read_csv(data_string)
    data['ds'] = pd.to_datetime(data['ds'])
    return data

# Custom Dataset class
class AWSCostDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Transformer Model
class AWSCostTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nhead=4, dropout=0.1):
        super().__init__()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, 
                                     dim_feedforward=hidden_dim*4, dropout=dropout),
            num_layers=num_layers
        )
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_embedding(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, hidden_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Back to (batch, seq_len, hidden_dim)
        x = self.output_layer(x[:, -1, :])  # Only use last sequence output
        return x

# Data preparation function
def prepare_data(data, sequence_length=7, test_days=7):
    # Split data by environment
    environments = data['unique_id'].unique()
    
    # Initialize scalers
    scalers = {}
    sequences = []
    targets = []
    
    for env in environments:
        env_data = data[data['unique_id'] == env]['y'].values
        
        # Create and fit scaler
        scaler = MinMaxScaler()
        env_data_scaled = scaler.fit_transform(env_data.reshape(-1, 1)).flatten()
        scalers[env] = scaler
        
        # Create sequences
        for i in range(len(env_data_scaled) - sequence_length - test_days):
            sequences.append(env_data_scaled[i:i+sequence_length])
            targets.append(env_data_scaled[i+sequence_length])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split into train and validation sets
    train_size = len(sequences) - (len(environments) * test_days)
    
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    val_sequences = sequences[train_size:]
    val_targets = targets[train_size:]
    
    return (train_sequences, train_targets, val_sequences, val_targets, 
            scalers, environments)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for sequences, targets in train_loader:
            sequences = sequences.float().to(device)
            targets = targets.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences.unsqueeze(-1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.float().to(device)
                targets = targets.float().to(device)
                
                outputs = model(sequences.unsqueeze(-1))
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Prediction and evaluation function
def evaluate_model(model, val_sequences, scalers, environments, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = {}
    actuals = {}
    
    with torch.no_grad():
        start_idx = 0
        for env in environments:
            env_data = data[data['unique_id'] == env]
            env_val_data = env_data.iloc[-7:]['y'].values
            
            env_sequences = torch.FloatTensor(
                val_sequences[start_idx:start_idx+7]).unsqueeze(-1).to(device)
            env_predictions = model(env_sequences).cpu().numpy()
            
            # Inverse transform predictions
            env_predictions = scalers[env].inverse_transform(
                env_predictions.reshape(-1, 1)).flatten()
            
            predictions[env] = env_predictions
            actuals[env] = env_val_data
            
            start_idx += 7
    
    return predictions, actuals

# Visualization functions
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(predictions, actuals, environments):
    plt.figure(figsize=(15, 5))
    
    for env in environments:
        plt.subplot(1, len(environments), environments.tolist().index(env) + 1)
        plt.plot(actuals[env], label='Actual', marker='o')
        plt.plot(predictions[env], label='Predicted', marker='s')
        plt.title(f'{env} Environment')
        plt.xlabel('Days')
        plt.ylabel('Cost ($)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(predictions, actuals, environments):
    metrics = {}
    
    for env in environments:
        env_metrics = {
            'RMSE': np.sqrt(mean_squared_error(actuals[env], predictions[env])),
            'MAE': mean_absolute_error(actuals[env], predictions[env]),
            'R2': r2_score(actuals[env], predictions[env])
        }
        metrics[env] = env_metrics
    
    return metrics

# Main execution
def main(data_string):
    # Load and prepare data
    data = load_data(data_string)
    (train_sequences, train_targets, val_sequences, val_targets, 
     scalers, environments) = prepare_data(data)
    
    # Create data loaders
    train_dataset = AWSCostDataset(train_sequences, train_targets)
    val_dataset = AWSCostDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and training components
    model = AWSCostTransformer(
        input_dim=1,
        hidden_dim=64,
        num_layers=3,
        output_dim=1
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer)
    
    # Evaluate model
    predictions, actuals = evaluate_model(model, val_sequences, scalers, 
                                        environments, data)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actuals, environments)
    
    # Visualizations
    plot_training_history(train_losses, val_losses)
    plot_predictions(predictions, actuals, environments)
    
    return metrics

# Example usage:
if __name__ == "__main__":
    # Your data string here
    data_string = "sample_data.csv"  # Your provided data
    
    metrics = main(data_string)
    print("\nEvaluation Metrics:")
    for env, env_metrics in metrics.items():
        print(f"\n{env} Environment:")
        for metric_name, value in env_metrics.items():
            print(f"{metric_name}: {value:.4f}")