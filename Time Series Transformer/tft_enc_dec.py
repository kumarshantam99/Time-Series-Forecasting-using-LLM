import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Read and preprocess data
def prepare_data(data_str):
    data = pd.read_csv(data_str)
    data['ds'] = pd.to_datetime(data['ds'])
    df_wide = data.pivot(index='ds', columns='unique_id', values='y')
    return df_wide

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length], 
                self.data[idx+1:idx+self.seq_length+1])

# Custom Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc(x)
        
        return x, attention

# Enhanced Transformer Encoder Layer
class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        # Embedding
        embedded = self.embedding(src)
        
        # Multi-head attention
        attention_output, attention_weights = self.attention(embedded, embedded, embedded, mask)
        attention_output = self.dropout(attention_output)
        
        # Add & Norm
        x = self.norm1(embedded + attention_output)
        
        # Feedforward
        ff_output = self.feedforward(x)
        ff_output = self.dropout(ff_output)
        
        # Add & Norm
        output = self.norm2(x + ff_output)
        
        return output, attention_weights

# Enhanced Transformer Model
class EnhancedTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim, dim_feedforward=512):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EnhancedTransformerEncoder(
                input_dim if i == 0 else d_model,
                d_model,
                num_heads,
                dim_feedforward
            ) for i in range(num_layers)
        ])
        self.decoder = nn.Linear(d_model, output_dim)
        self.attention_weights = None
        
    def forward(self, src):
        x = src
        attention_weights_layers = []
        
        for encoder in self.encoder_layers:
            x, attention_weights = encoder(x)
            attention_weights_layers.append(attention_weights)
        
        self.attention_weights = attention_weights_layers
        output = self.decoder(x)
        return output

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    attention_weights_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Store attention weights
            if epoch % 10 == 0:
                attention_weights_history.append([w.detach().cpu() for w in model.attention_weights])
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
    return train_losses, attention_weights_history

def plot_attention_weights(attention_weights, timestamps, save_path=None):
    """Plot attention weights for visualization"""
    plt.figure(figsize=(12, 8))
    
    # Get weights from the last layer, first head
    weights = attention_weights[-1][0].mean(0).numpy()
    
    sns.heatmap(weights, 
                xticklabels=timestamps,
                yticklabels=timestamps,
                cmap='viridis')
    
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Time Steps')
    plt.ylabel('Time Steps')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def run_forecasting(data_str):
    # Prepare data
    df = prepare_data(data_str)
    
    # Split data
    test_size = 7
    train_data = df.iloc[:-test_size]
    test_data = df.iloc[-test_size:]
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create datasets
    seq_length = 7
    train_dataset = TimeSeriesDataset(train_scaled, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # Initialize enhanced model
    input_dim = train_scaled.shape[1]
    d_model = 128
    num_heads = 8
    num_layers = 4
    output_dim = input_dim
    
    model = EnhancedTimeSeriesTransformer(
        input_dim, d_model, num_heads, num_layers, output_dim
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    num_epochs = 200
    train_losses, attention_weights_history = train_model(
        model, train_loader, criterion, optimizer, num_epochs
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_sequence = torch.FloatTensor(train_scaled[-seq_length:])
        predictions = []
        attention_weights = []
        
        for _ in range(test_size):
            output = model(test_sequence.unsqueeze(0))
            pred = output[0, -1].numpy()
            predictions.append(pred)
            attention_weights.append([w.cpu() for w in model.attention_weights])
            test_sequence = torch.cat((test_sequence[1:], torch.FloatTensor(pred).unsqueeze(0)))
    
    # Inverse transform predictions
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    
    # Calculate metrics
    metrics = {}
    for i, col in enumerate(df.columns):
        metrics[col] = calculate_metrics(test_data[col].values, predictions[:, i])
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 15))
    
    # Plot 1: Training Loss
    plt.subplot(3, 1, 1)
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot 2: Actual vs Predicted
    plt.subplot(3, 1, 2)
    for i, col in enumerate(df.columns):
        plt.plot(test_data.index, test_data[col].values, label=f'Actual {col}', linestyle='--')
        plt.plot(test_data.index, predictions[:, i], label=f'Predicted {col}')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 3: Attention Weights
    plt.subplot(3, 1, 3)
    last_attention = attention_weights[-1][0][0, 0].numpy()
    sns.heatmap(last_attention, cmap='viridis')
    plt.title('Attention Weights Heatmap (Last Prediction)')
    plt.xlabel('Source Sequence Position')
    plt.ylabel('Target Sequence Position')
    
    plt.tight_layout()
    
    return metrics, fig, attention_weights

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

data_str = 'sample_data.csv'
# Execute the forecasting
metrics, fig, attention_weights = run_forecasting(data_str)

# Print metrics
for environment, metric_values in metrics.items():
    print(f"\nMetrics for {environment}:")
    for metric_name, value in metric_values.items():
        print(f"{metric_name}: {value:.4f}")

plt.show()