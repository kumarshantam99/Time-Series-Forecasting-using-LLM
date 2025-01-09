import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from torch.optim.lr_scheduler import OneCycleLR

class TimesBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # FFN layers
        self.dropout = nn.Dropout(0.2)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, period: int) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        # Pad sequence if needed
        if seq_len % period != 0:
            pad_len = period - (seq_len % period)
            padding = torch.zeros(batch_size, pad_len, self.hidden_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = x.shape[1]
        
        # Reshape to (batch_size, num_periods, period, hidden_dim)
        x = x.reshape(batch_size, -1, period, self.hidden_dim)
        num_periods = x.shape[1]
        
        # Process each period independently
        for i in range(num_periods):
            period_data = x[:, i]  # (batch_size, period, hidden_dim)
            
            # Reshape for attention: (period, batch_size, hidden_dim)
            period_data = period_data.transpose(0, 1)
            
            # Self-attention
            attn_out, _ = self.attention(period_data, period_data, period_data)
            attn_out = attn_out.transpose(0, 1)  # (batch_size, period, hidden_dim)
            
            # First residual connection and normalization
            period_data = period_data.transpose(0, 1)
            period_data = self.norm1(period_data + attn_out)
            
            # FFN
            ffn_out = self.ffn(period_data)
            period_data = self.norm2(period_data + ffn_out)
            
            x[:, i] = period_data
        
        # Reshape back to original sequence length
        x = x.reshape(batch_size, -1, self.hidden_dim)
        if seq_len != x.shape[1]:
            x = x[:, :seq_len]
            
        return x

class TimesNet(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        periods: List[int] = [7, 14, 30]
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.periods = periods
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # TimesBlocks
        self.layers = nn.ModuleList([
            TimesBlock(input_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Adaptive pooling for variable length input
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Process through TimesBlocks with different periods
        period_outputs = []
        for period in self.periods:
            period_x = x
            for layer in self.layers:
                period_x = layer(period_x, period)
            period_outputs.append(period_x)
        
        # Average outputs from different periods
        x = torch.stack(period_outputs).mean(0)
        
        # Project to output dimension
        x = self.output_proj(x)  # (batch_size, seq_len, 1)
        
        # Reshape for adaptive pooling
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len)
        x = self.adaptive_pool(x)  # (batch_size, 1, output_len)
        x = x.transpose(1, 2)  # (batch_size, output_len, 1)
        
        return x

class AWSCostDataset(Dataset):
    def __init__(self, data: pd.DataFrame, team: str, seq_len: int, pred_len: int):
        team_data = data[data['Team'] == team]['Price'].values
        
        if len(team_data) < seq_len + pred_len:
            raise ValueError(f"Not enough data points for team {team}")
        
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(team_data.reshape(-1, 1))
        
        self.samples = []
        for i in range(len(team_data) - seq_len - pred_len + 1):
            x = self.scaled_data[i:i + seq_len]
            y = self.scaled_data[i + seq_len:i + seq_len + pred_len]
            self.samples.append((x, y))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

def train_model(
    data: pd.DataFrame,
    team: str,
    seq_len: int = 14,
    pred_len: int = 7,
    epochs: int = 500,
    batch_size: int = 8,
    learning_rate: float = 0.0001
):
    print(f"\nTraining model for {team}")
    print(f"Sequence length: {seq_len}")
    print(f"Prediction length: {pred_len}")
    
    # Create dataset
    dataset = AWSCostDataset(data, team, seq_len, pred_len)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = TimesNet(
        input_len=seq_len,
        output_len=pred_len,
        input_dim=1,
        hidden_dim=32,
        num_layers=2,
        periods=[7, 14]
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # Ensure correct input shape: (batch_size, seq_len, input_dim)
            if len(batch_x.shape) == 2:
                batch_x = batch_x.unsqueeze(-1)
            if len(batch_y.shape) == 2:
                batch_y = batch_y.unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset

def predict(
    model: TimesNet,
    dataset: AWSCostDataset,
    last_sequence: np.ndarray,
    pred_len: int
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # Ensure correct input shape
        x = torch.FloatTensor(last_sequence).unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        prediction = model(x)
        return dataset.inverse_transform(prediction.squeeze().numpy().reshape(-1, 1))

class ModelEvaluator:
    def __init__(self, true_values: np.ndarray, predictions: np.ndarray):
        self.true_values = true_values
        self.predictions = predictions
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate various evaluation metrics"""
        metrics = {
            'MAE': mean_absolute_error(self.true_values, self.predictions),
            'MSE': mean_squared_error(self.true_values, self.predictions),
            'RMSE': np.sqrt(mean_squared_error(self.true_values, self.predictions)),
            'R2': r2_score(self.true_values, self.predictions),
            'MAPE': np.mean(np.abs((self.true_values - self.predictions) / self.true_values)) * 100
        }
        return metrics

class TimeSeriesVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colors = {
            'Development': '#2ecc71',
            'Production': '#e74c3c',
            'Testing': '#3498db'
        }
        
    def plot_team_trends(self, figsize: Tuple[int, int] = (15, 8)) -> None:
        """Plot historical trends for all teams"""
        plt.figure(figsize=figsize)
        
        for team in self.df['Team'].unique():
            team_data = self.df[self.df['Team'] == team]
            plt.plot(team_data['Date'], team_data['Price'], 
                    label=team, color=self.colors[team])
            
        plt.title('AWS Cost Trends by Team', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cost ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def plot_forecast_comparison(self, 
                               team: str, 
                               true_values: np.ndarray, 
                               predictions: np.ndarray,
                               dates: pd.DatetimeIndex,
                               metrics: Dict[str, float],
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot actual vs predicted values with metrics"""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=(f'{team} - Forecast vs Actual',
                                         'Forecast Error Analysis'),
                           vertical_spacing=0.15)
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=dates, y=true_values.flatten(),
                      name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=predictions.flatten(),
                      name='Forecast', line=dict(color='red')),
            row=1, col=1
        )
        
        # Error Analysis
        errors = predictions.flatten() - true_values.flatten()
        fig.add_trace(
            go.Scatter(x=dates, y=errors,
                      name='Forecast Error',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # Add metrics annotation
        metrics_text = '<br>'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
        fig.add_annotation(
            xref='paper', yref='paper',
            x=1.0, y=1.0,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_layout(
            height=800,
            title_text=f"AWS Cost Forecast Analysis - {team}",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.show()
        
    def plot_seasonal_patterns(self, team: str, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Analyze and plot seasonal patterns"""
        team_data = self.df[self.df['Team'] == team].copy()
        team_data['DayOfWeek'] = team_data['Date'].dt.day_name()
        team_data['WeekOfMonth'] = team_data['Date'].dt.day // 7 + 1
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Daily patterns
        sns.boxplot(data=team_data, x='DayOfWeek', y='Price', 
                   order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   ax=ax1)
        ax1.set_title(f'{team} - Daily Cost Patterns')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Weekly patterns
        sns.boxplot(data=team_data, x='WeekOfMonth', y='Price', ax=ax2)
        ax2.set_title(f'{team} - Weekly Cost Patterns')
        ax2.set_xlabel('Week of Month')
        
        plt.tight_layout()
        plt.show()

def evaluate_and_visualize(
    df: pd.DataFrame,
    models: Dict[str, 'TimesNet'],
    datasets: Dict[str, 'AWSCostDataset'],
    validation_window: int = 7
) -> None:
    """Complete evaluation and visualization pipeline"""
    visualizer = TimeSeriesVisualizer(df)
    
    # Plot overall trends
    visualizer.plot_team_trends()
    
    for team in df['Team'].unique():
        # Get team data
        team_data = df[df['Team'] == team]['Price'].values
        
        # Get last sequence for validation
        val_start = -validation_window - 14  # 14 is sequence length
        last_sequence = datasets[team].scaler.transform(
            team_data[val_start:-validation_window].reshape(-1, 1)
        )
        
        # Make predictions
        predictions = predict(
            models[team],
            datasets[team],
            last_sequence,
            pred_len=validation_window
        )

        print(f"\nNext 7 days predictions for {team}:")
        print(predictions.flatten())
        
        # Get actual values for comparison
        actual_values = team_data[-validation_window:].reshape(-1, 1)
        
        # Calculate metrics
        evaluator = ModelEvaluator(actual_values, predictions)
        metrics = evaluator.calculate_metrics()
        
        # Get dates for validation period
        val_dates = df['Date'].unique()[-validation_window:]
        
        # Plot forecast comparison
        visualizer.plot_forecast_comparison(
            team=team,
            true_values=actual_values,
            predictions=predictions,
            dates=val_dates,
            metrics=metrics
        )
        
        # Plot seasonal patterns
        visualizer.plot_seasonal_patterns(team)

# Modified main function to include evaluation and visualization
def main():
    # Load data
   
    df = pd.read_csv("sample_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Train models
    teams = df['Team'].unique()
    models = {}
    datasets = {}
    
    for team in teams:
        print(f"\nTraining model for {team}...")
        model, dataset = train_model(df, team)
        models[team] = model
        datasets[team] = dataset
    
    # Evaluate and visualize results
    evaluate_and_visualize(df, models, datasets)

if __name__ == "__main__":
    main()