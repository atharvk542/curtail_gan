"""
Portfolio optimization and backtesting
Fixes Issues: 5, 24
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple
import warnings


def mean_variance_optimization(
    returns_scenarios: np.ndarray,
    target_return: float = None,
    risk_free_rate: float = 0.0001,
) -> Dict:
    """
    FIX Issue 5: Actual portfolio optimization using mean-variance framework.

    Solves: minimize portfolio_variance
            subject to: expected_return >= target_return
                       weights sum to 1
                       weights >= 0 (long-only)

    Args:
        returns_scenarios: Generated return scenarios, shape (n_scenarios,)
        target_return: Target portfolio return
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with optimal weights and statistics
    """
    returns_scenarios = returns_scenarios.flatten()

    # Portfolio: mix of risky asset (with generated returns) and risk-free asset
    # weight_risky in [0, 1], weight_rf = 1 - weight_risky

    mean_return = np.mean(returns_scenarios)
    var_return = np.var(returns_scenarios)

    if target_return is None:
        target_return = risk_free_rate + 0.5 * (mean_return - risk_free_rate)

    # Analytical solution for two-asset case
    if mean_return <= risk_free_rate:
        # Risky asset has negative Sharpe, invest 100% in risk-free
        optimal_weight_risky = 0.0
        warnings.warn(
            "Risky asset has negative Sharpe ratio. 100% risk-free allocation."
        )
    else:
        # Target return: w * mean_return + (1-w) * rf_rate >= target_return
        # Solve for w
        if abs(mean_return - risk_free_rate) < 1e-10:
            optimal_weight_risky = 0.5  # Indifferent
        else:
            optimal_weight_risky = (target_return - risk_free_rate) / (
                mean_return - risk_free_rate
            )
            optimal_weight_risky = np.clip(optimal_weight_risky, 0.0, 1.0)

    # Portfolio statistics
    portfolio_return = (
        optimal_weight_risky * mean_return + (1 - optimal_weight_risky) * risk_free_rate
    )
    portfolio_variance = (optimal_weight_risky**2) * var_return
    portfolio_std = np.sqrt(portfolio_variance)

    # Sharpe ratio
    if portfolio_std > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    else:
        sharpe_ratio = 0.0

    return {
        "weight_risky": optimal_weight_risky,
        "weight_risk_free": 1 - optimal_weight_risky,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_std,
        "sharpe_ratio": sharpe_ratio,
        "target_return": target_return,
    }


def cvar_optimization(
    returns_scenarios: np.ndarray,
    confidence_level: float = 0.95,
    target_return: float = None,
    risk_free_rate: float = 0.0001,
) -> Dict:
    """
    FIX Issue 24: CVaR (Conditional Value at Risk) minimization.

    More appropriate for tail risk management.

    Args:
        returns_scenarios: Generated return scenarios
        confidence_level: CVaR confidence level
        target_return: Target portfolio return
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with optimal weights and statistics
    """
    returns_scenarios = returns_scenarios.flatten()

    def compute_cvar(weight_risky, scenarios, alpha):
        """Compute CVaR for given weight"""
        portfolio_returns = (
            weight_risky * scenarios + (1 - weight_risky) * risk_free_rate
        )
        var = np.percentile(portfolio_returns, (1 - alpha) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return -cvar  # Negative because we minimize

    mean_return = np.mean(returns_scenarios)

    if target_return is None:
        target_return = risk_free_rate + 0.3 * (mean_return - risk_free_rate)

    # Constraint: portfolio return >= target
    def return_constraint(weight_risky):
        port_ret = weight_risky * mean_return + (1 - weight_risky) * risk_free_rate
        return port_ret - target_return

    # Optimize
    result = minimize(
        fun=lambda w: compute_cvar(w[0], returns_scenarios, confidence_level),
        x0=[0.5],
        bounds=[(0, 1)],
        constraints={"type": "ineq", "fun": lambda w: return_constraint(w[0])},
        method="SLSQP",
    )

    if result.success:
        optimal_weight_risky = result.x[0]
    else:
        warnings.warn("CVaR optimization failed. Using mean-variance.")
        mv_result = mean_variance_optimization(
            returns_scenarios, target_return, risk_free_rate
        )
        optimal_weight_risky = mv_result["weight_risky"]

    # Compute portfolio stats
    portfolio_returns = (
        optimal_weight_risky * returns_scenarios
        + (1 - optimal_weight_risky) * risk_free_rate
    )
    portfolio_return = np.mean(portfolio_returns)
    portfolio_std = np.std(portfolio_returns)

    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    sharpe = (
        (portfolio_return - risk_free_rate) / portfolio_std
        if portfolio_std > 0
        else 0.0
    )

    return {
        "weight_risky": optimal_weight_risky,
        "weight_risk_free": 1 - optimal_weight_risky,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_std,
        "sharpe_ratio": sharpe,
        "var": var,
        "cvar": cvar,
        "confidence_level": confidence_level,
    }


def backtest_portfolio(
    test_returns: np.ndarray, weight_risky: float, risk_free_rate: float = 0.0001
) -> Dict:
    """
    FIX Issue 24: Backtest portfolio on held-out data.

    Args:
        test_returns: Actual returns in test period
        weight_risky: Weight allocated to risky asset
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with performance metrics
    """
    test_returns = test_returns.flatten()

    # Portfolio returns
    portfolio_returns = (
        weight_risky * test_returns + (1 - weight_risky) * risk_free_rate
    )

    # Cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns)

    # Performance metrics
    total_return = cumulative_returns[-1] - 1
    annualized_return = (1 + total_return) ** (
        52 / len(test_returns)
    ) - 1  # Assuming weekly

    portfolio_vol = np.std(portfolio_returns) * np.sqrt(52)  # Annualized
    sharpe = (
        (annualized_return - risk_free_rate * 52) / portfolio_vol
        if portfolio_vol > 0
        else 0.0
    )

    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": portfolio_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "cumulative_returns": cumulative_returns,
    }


def compare_portfolio_strategies(
    baseline_scenarios: np.ndarray,
    curritail_scenarios: np.ndarray,
    test_returns: np.ndarray,
    target_return: float = 0.0005,
    risk_free_rate: float = 0.0001,
    use_cvar: bool = True,
) -> Dict:
    """
    FIX Issue 5 & 24: Complete portfolio comparison pipeline.

    1. Optimize weights using each model's generated scenarios
    2. Backtest on held-out real data
    3. Compare performance

    Args:
        baseline_scenarios: Scenarios from baseline model
        curritail_scenarios: Scenarios from CurriTail model
        test_returns: Held-out real returns for backtesting
        target_return: Target portfolio return
        risk_free_rate: Risk-free rate
        use_cvar: Use CVaR optimization instead of mean-variance

    Returns:
        Dictionary with comparison results
    """
    optimize_func = cvar_optimization if use_cvar else mean_variance_optimization

    # Optimize using each model
    baseline_opt = optimize_func(
        baseline_scenarios, target_return=target_return, risk_free_rate=risk_free_rate
    )
    curritail_opt = optimize_func(
        curritail_scenarios, target_return=target_return, risk_free_rate=risk_free_rate
    )

    # Backtest on real data
    baseline_backtest = backtest_portfolio(
        test_returns, baseline_opt["weight_risky"], risk_free_rate
    )
    curritail_backtest = backtest_portfolio(
        test_returns, curritail_opt["weight_risky"], risk_free_rate
    )

    # Compare
    comparison = {
        "baseline_optimization": baseline_opt,
        "curritail_optimization": curritail_opt,
        "baseline_backtest": baseline_backtest,
        "curritail_backtest": curritail_backtest,
        "improvement": {
            "sharpe_diff": curritail_backtest["sharpe_ratio"]
            - baseline_backtest["sharpe_ratio"],
            "return_diff": curritail_backtest["annualized_return"]
            - baseline_backtest["annualized_return"],
            "drawdown_diff": curritail_backtest["max_drawdown"]
            - baseline_backtest["max_drawdown"],  # Negative is better
            "vol_diff": curritail_backtest["annualized_volatility"]
            - baseline_backtest["annualized_volatility"],
        },
    }

    return comparison


def simulate_crash_scenario(
    n_days: int = 252,
    crash_start: int = 200,
    crash_magnitude: float = -0.05,
    crash_duration: int = 5,
    normal_mean: float = 0.0005,
    normal_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate market returns with embedded crash for Figure 6.

    Args:
        n_days: Total trading days
        crash_start: Day when crash starts
        crash_magnitude: Daily return during crash
        crash_duration: Number of crash days
        normal_mean: Normal period daily return
        normal_std: Normal period volatility
        seed: Random seed

    Returns:
        Array of simulated returns
    """
    np.random.seed(seed)

    # Normal periods
    normal_returns = np.random.normal(normal_mean, normal_std, n_days)

    # Insert crash
    crash_end = crash_start + crash_duration
    crash_returns = np.random.normal(crash_magnitude, normal_std * 1.5, crash_duration)
    normal_returns[crash_start:crash_end] = crash_returns

    return normal_returns
