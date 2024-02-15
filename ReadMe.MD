# Portfolio Management System

This portfolio management system optimizes a stock portfolio, evaluates various financial performance metrics, and visualizes performance over time and against benchmarks. The system employs modular design, with scripts dedicated to fetching data, optimization, performance evaluation, and plotting.

## Why Monte Carlo Simulations and Sharpe Ratio Optimizations?

- **Monte Carlo Simulations**: Used to explore a wide range of possible future outcomes by allowing for random variations in asset returns. This approach helps in understanding the potential variability in portfolio performance, making it valuable for assessing risk and return characteristics under different scenarios.

- **Sharpe Ratio Optimizations**: A method to maximize the Sharpe Ratio, which measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. This optimization helps in identifying the portfolio allocation that offers the best risk-adjusted return, making it a cornerstone for portfolio optimization.

## Performance Metrics

- **Sharpe Ratio**: Measures excess return per unit of risk, helping investors understand the return of an investment compared to its risk.

- **Annualized Return**: The yearly compounded return of the portfolio, providing a standardized measure of returns over time.

- **Annualized Volatility**: Represents the standard deviation of annualized returns, a measure of the portfolio's risk or variability.

- **Treynor Measure**: Similar to the Sharpe Ratio but uses beta (a measure of market risk) instead of total risk, focusing on returns earned in excess of that which could have been earned on a riskless investment per each unit of market risk.

- **Jensen's Alpha**: Measures the average return of the portfolio over the expected return based on the CAPM model, indicating the portfolio manager's ability to generate excess returns.

- **Value at Risk (VaR)**: Estimates the maximum potential loss of the portfolio over a specified time frame at a given confidence interval.

- **Conditional Value at Risk (CVaR)**: Provides the expected loss exceeding the VaR, offering insight into the tail risk of the portfolio.

## Installation

To set up the portfolio management system:

1. Ensure Python is installed on your system.
2. Install necessary Python packages using pip:

```python
pip install pandas numpy matplotlib scipy yfinance
```

3. Clone the repository or download the scripts to your local machine.
4. Run main.py to initiate the system workflow:

```python
main.py
```
## Future Developments

- **Portfolio Rebalancing**: Future versions will incorporate a dynamic rebalancing feature, allowing the system to automatically adjust the portfolio weights based on updated market conditions and optimization results. This will ensure the portfolio maintains its desired risk-return profile over time, especially important as the code will be run monthly to identify optimal stock weights.

- **Enhanced Forecasting Techniques**: Integrating more sophisticated forecasting models to improve the accuracy of future performance predictions.

- **Real-Time Data Integration**: Allowing the system to fetch and process real-time market data for up-to-the-minute portfolio optimization and analysis.

## Usage

Modify `main.py` as needed to specify stock symbols, the date range, and any parameters for optimization and metric calculations. This script coordinates the workflow, fetching data, performing optimizations, calculating metrics, and generating plots.

## Notes

The system's modular design facilitates easy updates or modifications to individual components, accommodating evolving investment strategies or analysis techniques.

## Author
[Hisho Rajanathan]