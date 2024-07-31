from typing import List
from return_calculator import ReturnCalculator
from plotter import Plotter


def main() -> None:
    tickers: List[str] = ["^NDX"] #, "^GSPC"]  # Nasdaq-100 and S&P 500 index tickers
    start_date: str = '1980-01-01'  # Adjust the start date as needed

    return_calculator = ReturnCalculator(tickers, start_date)
    plotter = Plotter(return_calculator)

    # Plot quarterly returns for all tickers
    plotter.get_median_quarterly_returns()

if __name__ == "__main__":
    main()
