from typing import List
from return_calculator import ReturnCalculator
from plotter import Plotter
import numpy as np

import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy

def main() -> None:
    # tickers: List[str] = ["^NDX"] # Nasdaq-100 index
    # tickers: List[str] = ["BTC-USD"]  # Bitcoin
    tickers: List[str] = ["^SPX"]  # S&P 500 index
    # tickers: List[str] = ["^RUT"] # Russell 2000 index
    start_date: str = '1990-01-01'  # Adjust the start date as needed

    return_calculator = ReturnCalculator(tickers, start_date)
    plotter = Plotter(return_calculator)

    # Plot quarterly returns for all tickers
    # plotter.get_average_quarterly_returns()
    # plotter.get_average_monthly_returns()
    # plotter.get_average_semi_monthly_returns()
    # plotter.get_average_monthly_volatility()
    plotter.plot_price_sma_difference()
    # plotter.plot_sma_first_derivative()

if __name__ == "__main__":
    main()
