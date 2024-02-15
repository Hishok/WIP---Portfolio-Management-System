import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_stock_performance(data):
    """
    Plots the performance of individual stocks over time.

    Parameters:
    - data: DataFrame containing stock prices with dates as index and tickers as columns.
    """
    plt.figure(figsize=(14, 7))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.title('Stock Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.show()

def plot_weight_changes(weights_data):
    """
    Plots how portfolio weights change over time.

    Parameters:
    - weights_data: DataFrame containing portfolio weights over time with dates
    as index and tickers as columns.
    """
    weights_data.plot(figsize=(14, 7))
    plt.title('Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(title='Ticker')
    plt.show()

def plot_portfolio_performance(portfolio_returns, benchmark_returns):
    """
    Plots the portfolio performance against benchmarks over time.

    Parameters:
    - portfolio_returns: Series containing portfolio returns with dates as index.
    - benchmark_returns: DataFrame containing benchmark returns with dates as index 
    and benchmark names as columns.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_returns.index, np.cumprod(1 + portfolio_returns), label='Portfolio', color='black')
    for column in benchmark_returns.columns:
        plt.plot(benchmark_returns.index, np.cumprod(1 + benchmark_returns[column]), label=column)
    plt.title('Portfolio Performance vs. Benchmarks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def plot_future_forecast(base_returns, forecasted_returns):
    """
    Plots a forecast of future portfolio performance based on simulated returns.

    Parameters:
    - base_returns: Series containing historical portfolio returns for the base period.
    - forecasted_returns: DataFrame containing forecasted returns for multiple scenarios.
    """
    plt.figure(figsize=(14, 7))
    # Plot historical performance
    plt.plot(base_returns.index, np.cumprod(1 + base_returns), label='Historical Performance', color='blue')
    # Plot forecasted performance scenarios
    for scenario in forecasted_returns.columns:
        plt.plot(forecasted_returns.index, np.cumprod(1 + forecasted_returns[scenario]), linestyle='--', alpha=0.5)
    plt.title('Forecasted Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()
