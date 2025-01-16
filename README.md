# Time-Series-Forecasting-using-LLM
Experimentations on performance of LLM models on Time Series forecasting

## Time series Transformer

A Time Series Transformer is an adaptation of the original Transformer architecture (initially designed for natural language processing) specifically optimized for time series data. Let me break this down:
Key Components and Characteristics:

* Input Encoding: Time series data is converted into sequences of tokens, similar to how words are tokenized in NLP
Typically includes both the actual values and temporal information (timestamps, seasonality, etc.)
Often incorporates positional encodings to maintain temporal order

* Self-Attention Mechanism: Allows the model to weigh the importance of different timestamps
Can capture both long-term and short-term dependencies in the data
Particularly effective at identifying patterns across different time scales

* Key Adaptations for Time Series: Causal attention masks to prevent looking at future values during training
Special embedding layers for temporal features Modified positional encodings to handle irregular time intervals
Often includes convolutional layers to capture local patterns.

