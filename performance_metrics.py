import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_annualized_return(returns):
    """
    Calculates the annualized return of a portfolio.
    """
    compounded_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    return compounded_return

def calculate_annualized_volatility(returns):
    """
    Calculates the annualized volatility of a portfolio.
    """
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculates the Sharpe Ratio of a portfolio. - A measure for calculating
    risk-adjusted return. It represents the additional amount of return that an 
    investor receives per unit of increase in risk.
    """
    annualized_return = calculate_annualized_return(returns)
    annualized_volatility = calculate_annualized_volatility(returns)
    return (annualized_return - risk_free_rate) / annualized_volatility

def calculate_treynor_measure(returns, beta, risk_free_rate=0.01):
    """
    Calculates the Treynor Measure of a portfolio. - Similar to the Sharpe ratio,
    but uses beta (market risk) instead of total risk (volatility).
    """
    annualized_return = calculate_annualized_return(returns)
    return (annualized_return - risk_free_rate) / beta

def calculate_jensens_alpha(returns, market_returns, beta, risk_free_rate=0.01):
    """
    Calculates Jensen's Alpha of a portfolio. - A measure of the average return on a 
    portfolio or investment above or below that predicted by the CAPM, given the 
    portfolio's or investment's beta and the average market return.
    """
    annualized_return = calculate_annualized_return(returns)
    annualized_market_return = calculate_annualized_return(market_returns)
    expected_return = risk_free_rate + beta * (annualized_market_return - risk_free_rate)
    return annualized_return - expected_return

def calculate_downside_risk(returns, target=0):
    """
    Calculates the downside risk of a portfolio. - Focuses on returns that 
    fall below a minimum threshold or target, typically zero or the risk-free rate,
    capturing risk in negative returns.
    """
    downside_returns = returns[returns < target]
    return downside_returns.std() * np.sqrt(252)

def calculate_var(returns, confidence_level=0.05):
    """
    Calculates the Value at Risk (VaR) of a portfolio at a given confidence level. - A 
    statistical technique used to measure and quantify the level of financial risk 
    within a firm, portfolio, or position over a specific time frame.
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculates the Conditional Value at Risk (CVaR) of a portfolio at a given confidence level.
    Provides the expected loss in the worst-case scenario of investment (beyond the VaR level).
    """
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar
