import return_calculator as ReturnCalculator
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
# Set the backend to a non-inline backend that supports interactive windows
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your preference


class Plotter:
    def __init__(self, return_calculator: ReturnCalculator) -> None:
        self.return_calculator = return_calculator



    def get_median_quarterly_returns(self) -> None:
        """Plot median quarterly returns for every 4-year cycle leading up to an election year."""
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        cycle_aggregated_returns = {cycle: {q: [] for q in quarters} for cycle in ['n-0', 'n-1', 'n-2', 'n-3']}

        # Collect all returns for each quarter from each ticker
        for ticker in self.return_calculator.tickers:
            median_returns = self.return_calculator.calculate_median_quarterly_returns_cycle(ticker)
            if not median_returns:
                print(f"No data available for ticker {ticker}")
                continue

            for cycle in cycle_aggregated_returns:
                for quarter in quarters:
                    cycle_aggregated_returns[cycle][quarter].extend(median_returns.get(cycle, {}).get(quarter, []))

        # Calculate the final median returns
        final_median_returns = {cycle: {} for cycle in cycle_aggregated_returns}
        for cycle, quarters in cycle_aggregated_returns.items():
            for quarter, values in quarters.items():
                if values:
                    final_median_returns[cycle][quarter] = np.median(values)
                else:
                    final_median_returns[cycle][quarter] = None

        return self.plot_median_quarterly_returns(final_median_returns)

    def plot_median_quarterly_returns(self, final_median_returns):
        """Plot median quarterly returns for every 4-year cycle leading up to an election year."""
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        cycles = ['n-3', 'n-2', 'n-1', 'n-0']

        # Prepare data for plotting
        values = [final_median_returns.get(cycle, {}).get(q, 0) for cycle in cycles for q in quarters]

        n_bars = len(quarters)
        n_cycles = len(cycles)
        x = np.arange(n_cycles * n_bars)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.8
        bars = ax.bar(x, values, width=bar_width, alpha=0.7, label='Median Return')

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([q for _ in cycles for q in quarters], rotation=45)
        cycles = ['n-3', 'n-2', 'n-1', 'Election Year (n)']
        for i in range(1, n_cycles):
            ax.axvline(x[i * n_bars] - 0.5, color='grey', linestyle='--')

        for i, cycle in enumerate(cycles):
            position = (i * n_bars) + n_bars / 2 - 0.5
            ax.text(position, -0.1, cycle, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)

        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        ax.set_xlim([x[0] - 0.5, x[-1] + 0.5])
        ax.set_title('NQ Median Quarterly Returns by Election Cycle')
        ax.set_ylabel('Median Return (%)')
        plt.legend()
        plt.tight_layout()
        plt.show()


