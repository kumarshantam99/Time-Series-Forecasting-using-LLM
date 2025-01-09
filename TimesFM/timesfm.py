import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error
)
from scipy import stats

# Attempt multiple import strategies
try:
    from transformers import TimesFmForPrediction
except ImportError:
    try:
        from transformers import TimeSeriesForPrediction as TimesFmForPrediction
    except ImportError:
        try:
            from transformers.models.timesfm import TimesFmForPrediction
        except ImportError:
            TimesFmForPrediction = None
            print("Could not import TimesFM model. Please check your Transformers library version.")

class TimeSeriesForecaster:
    def __init__(self, model_name='google/timesfm-1.0-200m-pytorch'):
        """
        Initialize TimesFM forecaster
        
        Args:
        model_name (str): Hugging Face model identifier
        """
        if TimesFmForPrediction is None:
            raise ImportError("TimesFM model could not be imported. Please verify your setup.")
        
        try:
            self.model = TimesFmForPrediction.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Possible causes:")
            print("1. Check internet connection")
            print("2. Verify model name")
            print("3. Ensure you have sufficient permissions")
            raise
        
    def prepare_data(self, df, team, context_len=7, horizon_len=7):
        """
        Prepare time series data for forecasting
        
        Args:
        df (pd.DataFrame): Input dataframe
        team (str): Team to forecast
        context_len (int): Historical context length
        horizon_len (int): Forecast horizon
        
        Returns:
        dict: Prepared forecast inputs
        """
        # Filter and sort data for specific team
        team_data = df[df['Team'] == team].sort_values('Date')
        prices = team_data['Price'].values
        
        # Ensure sufficient historical data
        if len(prices) < context_len:
            raise ValueError(f"Insufficient data for {team}. Need at least {context_len} points.")
        
        # Prepare input and split for validation
        input_prices = prices[-context_len:]
        input_ids = torch.tensor(input_prices, dtype=torch.float32).unsqueeze(0)
        
        return {
            'input_ids': input_ids,
            'forecast_horizon': horizon_len,
            'historical_data': team_data
        }
    
    def forecast(self, data):
        """
        Generate forecasts
        
        Args:
        data (dict): Prepared forecast inputs
        
        Returns:
        dict: Forecast results
        """
        outputs = self.model.forecast(
            input_ids=data['input_ids'], 
            forecast_horizon=data['forecast_horizon']
        )
        
        return {
            'point_forecast': outputs.prediction.numpy()[0],
            'quantile_forecasts': outputs.quantile_predictions.numpy()[0]
        }
    
    def evaluate_forecast(self, historical_data, forecasts, horizon_len=7):
        """
        Compute forecast performance metrics
        
        Args:
        historical_data (pd.DataFrame): Historical time series data
        forecasts (dict): Generated forecasts
        horizon_len (int): Forecast horizon
        
        Returns:
        dict: Performance metrics
        """
        # Actual future values (for last 7 days)
        actual_values = historical_data['Price'].values[-horizon_len:]
        point_forecast = forecasts['point_forecast']
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(actual_values, point_forecast),
            'MSE': mean_squared_error(actual_values, point_forecast),
            'RMSE': np.sqrt(mean_squared_error(actual_values, point_forecast)),
            'MAPE': mean_absolute_percentage_error(actual_values, point_forecast),
            'R2': stats.pearsonr(actual_values, point_forecast)[0] ** 2
        }
        
        return metrics
    
    def visualize_forecast(self, historical_data, forecasts, team):
        """
        Create comprehensive visualization of forecast
        
        Args:
        historical_data (pd.DataFrame): Historical time series data
        forecasts (dict): Generated forecasts
        team (str): Team name
        """
        plt.figure(figsize=(15, 10))
        
        # Historical data
        plt.subplot(2, 1, 1)
        plt.plot(historical_data['Date'], historical_data['Price'], label='Historical')
        plt.title(f'{team} - Historical Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        
        # Forecast with uncertainty
        plt.subplot(2, 1, 2)
        historical_prices = historical_data['Price'].values
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=historical_data['Date'].max() + pd.Timedelta(days=1), 
            periods=len(forecasts['point_forecast'])
        )
        
        # Plot point forecast
        plt.plot(forecast_dates, forecasts['point_forecast'], 
                 label='Point Forecast', color='red')
        
        # Plot quantile bands
        plt.fill_between(
            forecast_dates, 
            forecasts['quantile_forecasts'][:, 0],  # Lower quantile
            forecasts['quantile_forecasts'][:, -1],  # Upper quantile
            alpha=0.2, 
            color='red', 
            label='Forecast Interval'
        )
        
        plt.title(f'{team} - Forecast with Uncertainty')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_forecast(self, df, team):
        """
        Complete forecasting pipeline
        
        Args:
        df (pd.DataFrame): Input dataframe
        team (str): Team to forecast
        
        Returns:
        dict: Forecast results and metrics
        """
        # Prepare data
        data = self.prepare_data(df, team)
        
        # Generate forecast
        forecasts = self.forecast(data)
        
        # Evaluate forecast
        metrics = self.evaluate_forecast(
            data['historical_data'], 
            forecasts
        )
        
        # Visualize results
        self.visualize_forecast(
            data['historical_data'], 
            forecasts, 
            team
        )
        
        return {
            'forecasts': forecasts,
            'metrics': metrics
        }

def main():
    # Simulated sample data generation (since actual CSV is not available)
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100)
    teams = ['Development', 'Production', 'Testing']
    
    sample_data = []
    for team in teams:
        base_price = np.random.randint(100, 500)
        trend = np.random.uniform(-0.5, 0.5)
        team_prices = base_price + trend * np.arange(len(dates)) + np.random.normal(0, 10, len(dates))
        team_df = pd.DataFrame({
            'Date': dates,
            'Team': team,
            'Price': team_prices
        })
        sample_data.append(team_df)
    
    df = pd.concat(sample_data, ignore_index=True)
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster()
    
    # Store results
    all_results = {}
    
    # Forecast for each team
    for team in teams:
        print(f"\nForecasting for {team} Team:")
        try:
            results = forecaster.run_forecast(df, team)
            all_results[team] = results
            
            # Print metrics
            print("\nForecast Metrics:")
            for metric, value in results['metrics'].items():
                print(f"{metric}: {value:.4f}")
        
        except Exception as e:
            print(f"Error forecasting for {team}: {e}")
    
    # Comparative Performance Visualization
    plt.figure(figsize=(12, 6))
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [all_results[team]['metrics'][metric] for team in teams]
        plt.bar(teams, values)
        plt.title(f'{metric} by Team')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()