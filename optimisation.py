import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculates portfolio performance metrics including annualized return,
    annualized volatility, and Sharpe Ratio.
    """
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def monte_carlo_simulation(returns, num_simulations=10000, risk_free_rate=0.01):
    """
    Performs portfolio optimization using Monte Carlo simulation to find
    the portfolio with the highest Sharpe Ratio.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    results = np.zeros((3, num_simulations))
    weight_array = []

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_array.append(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_allocation = weight_array[max_sharpe_idx]
    
    return max_sharpe_allocation, results[:, max_sharpe_idx]

def optimize_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Directly optimizes the portfolio for the highest Sharpe Ratio using scipy's minimize function.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def neg_sharpe(weights): 
        return -calculate_portfolio_performance(weights, *args)[2]

    result = minimize(neg_sharpe, num_assets*[1./num_assets,], bounds=bounds, constraints=constraints)
    
    return result.x

if __name__ == "__main__":
    # Example usage with hypothetical returns DataFrame
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, size=(252, 4)))  # Mock data
    max_sharpe_allocation_mc, mc_results = monte_carlo_simulation(returns)
    optimized_weights_sr = optimize_sharpe_ratio(returns)
    
    print(f"Optimized Weights (Monte Carlo): {max_sharpe_allocation_mc}")
    print(f"Optimized Weights (Sharpe Ratio Optimization): {optimized_weights_sr}")
