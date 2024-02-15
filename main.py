from datetime import datetime
from data_fetch import fetch_historical_prices
from optimisation import monte_carlo_simulation, optimize_sharpe_ratio
from performance_metrics import (calculate_sharpe_ratio, calculate_annualized_return, 
                                 calculate_annualized_volatility, calculate_treynor_measure, 
                                 calculate_jensens_alpha, calculate_var, calculate_cvar)
from plotting import (plot_stock_performance, plot_weight_changes, 
                      plot_portfolio_performance, plot_future_forecast)

def main():
    # Define symbols and date range
    symbols = ['AAPL', 'VUSA.L', 'VWRL.L', 'LLOY.L', 'SHOP', 'NVDA', 'SMT.L',
               'BP.L', 'TSLA', 'DIS', 'KO', 'GOOGL', 'MCD', 'AXP', 'AMD',
               'SBUX', 'T', 'AV.L', 'PEP', 'BARC.L', 'PYPL', 'NFLX',
               'AMZN', 'COIN', 'MSFT', 'ORCL', 'META', 'CRM', 'IBM']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch historical data
    historical_prices = fetch_historical_prices(symbols, start_date, end_date)
    
    # Plot stock performance
    plot_stock_performance(historical_prices)

    # Perform optimization using Monte Carlo simulation and Sharpe Ratio optimization
    returns = historical_prices.pct_change().dropna()
    optimal_weights_mc, mc_results = monte_carlo_simulation(returns)
    optimal_weights_sr = optimize_sharpe_ratio(returns)
    print(f"Optimal Portfolio Weights (Monte Carlo): {optimal_weights_mc}")
    print(f"Optimal Portfolio Weights (Sharpe Ratio Optimization): {optimal_weights_sr}")

    # Calculate and display performance metrics
    portfolio_returns_mc = returns.dot(optimal_weights_mc)
    portfolio_returns_sr = returns.dot(optimal_weights_sr)
    # Calculate metrics for Monte Carlo optimized portfolio
    sharpe_ratio_mc = calculate_sharpe_ratio(portfolio_returns_mc)
    annualized_return_mc = calculate_annualized_return(portfolio_returns_mc)
    annualized_volatility_mc = calculate_annualized_volatility(portfolio_returns_mc)
    # Example: using VWRL.L as a benchmark for Monte Carlo optimized portfolio
    benchmark_returns = returns['VWRL.L']
    beta_mc = 1  # 
    treynor_measure_mc = calculate_treynor_measure(portfolio_returns_mc, beta_mc)
    jensens_alpha_mc = calculate_jensens_alpha(portfolio_returns_mc, benchmark_returns, beta_mc)
    var_mc = calculate_var(portfolio_returns_mc)
    cvar_mc = calculate_cvar(portfolio_returns_mc)

    print(f"Monte Carlo Optimized Portfolio Performance Metrics:")
    print(f"Sharpe Ratio: {sharpe_ratio_mc}")
    print(f"Annualized Return: {annualized_return_mc}")
    print(f"Annualized Volatility: {annualized_volatility_mc}")
    print(f"Treynor Measure: {treynor_measure_mc}")
    print(f"Jensen's Alpha: {jensens_alpha_mc}")
    print(f"Value at Risk (VaR): {var_mc}")
    print(f"Conditional Value at Risk (CVaR): {cvar_mc}")

    # Plotting
    # Note: Implement plot_weight_changes and plot_future_forecast with your data.
    benchmarks_data = fetch_historical_prices(['VWRL.L', 'SPY'], start_date, end_date).pct_change().dropna()
    plot_portfolio_performance(portfolio_returns_mc.cumsum(), benchmarks_data.cumsum())

if __name__ == "__main__":
    main()
