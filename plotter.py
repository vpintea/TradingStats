import return_calculator as ReturnCalculator
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Set the backend to a non-inline backend that supports interactive windows
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your preference

class Plotter:
    def __init__(self, return_calculator: ReturnCalculator) -> None:
        self.return_calculator = return_calculator

    def get_average_quarterly_returns(self) -> None:
        """Plot average quarterly returns for every 4-year cycle leading up to an election year."""
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        cycle_aggregated_returns = {cycle: {q: [] for q in quarters} for cycle in ['n-0', 'n-1', 'n-2', 'n-3']}

        # Collect all returns for each quarter from each ticker
        for ticker in self.return_calculator.tickers:
            average_returns = self.return_calculator.calculate_average_quarterly_returns_cycle(ticker)
            if not average_returns:
                print(f"No data available for ticker {ticker}")
                continue

            for cycle in cycle_aggregated_returns:
                for quarter in quarters:
                    cycle_aggregated_returns[cycle][quarter].extend(average_returns.get(cycle, {}).get(quarter, []))

        # Calculate the final average returns
        final_average_returns = {cycle: {} for cycle in cycle_aggregated_returns}
        for cycle, quarters in cycle_aggregated_returns.items():
            for quarter, values in quarters.items():
                if values:
                    final_average_returns[cycle][quarter] = np.average(values)
                else:
                    final_average_returns[cycle][quarter] = None

        return self.plot_average_quarterly_returns(final_average_returns)

    def plot_average_quarterly_returns(self, final_average_returns):
        """Plot average quarterly returns for every 4-year cycle leading up to an election year."""
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        cycles = ['n-3', 'n-2', 'n-1', 'n-0']

        # Prepare data for plotting
        values = [final_average_returns.get(cycle, {}).get(q, 0) for cycle in cycles for q in quarters]

        n_bars = len(quarters)
        n_cycles = len(cycles)
        bar_width = 0.8
        # positions for each bar, adjusted for bar width
        x = np.arange(len(values)) * (bar_width + 0.1) + bar_width / 2

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, values, width=bar_width, alpha=0.7, label='Average Return')

        # Set x-ticks to be in the middle of each bar
        ax.set_xticks(x)
        ax.set_xticklabels([q for _ in cycles for q in quarters], rotation=45)
        cycles = ['n-3', 'n-2', 'n-1', 'Election Year (n)']

        # vertical lines to separate cycles
        for i in range(1, n_cycles):
            ax.axvline(x[i * n_bars] - 0.45, color='grey', linestyle='--')

        # 2nd axis
        for i, cycle in enumerate(cycles):
            position = (i * n_bars) + n_bars / 2 - 1.0
            ax.text(position, -0.1, cycle, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)

        # % annotation on top of bars
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        ax.set_xlim([x[0] - 0.5, x[-1] + 0.5])
        date = self.return_calculator.earliest_data_year
        ticker = self.return_calculator.tickers
        title = str(ticker[0].strip()) + " Average Quarterly Returns by Election Cycle since " + str(date)
        ax.set_title(title)
        ax.set_ylabel('Average Return (%)')
        plt.legend()
        plt.tight_layout()
        file_path: str = f'Quarterly Charts/'
        plt.savefig(file_path + title + ".png")
        plt.show()

    def get_average_monthly_returns(self):
        """Retrieve average monthly returns for all tickers and prepare for plotting."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_aggregated_returns = {month: [] for month in months}

        for ticker in self.return_calculator.tickers:
            monthly_returns = self.return_calculator.calculate_monthly_returns(ticker)
            if monthly_returns.empty:  # Correct way to check if a DataFrame or Series is empty
                print(f"No data available for ticker {ticker}")
                continue

            for month in range(1, 13):  # Month indices from 1 to 12
                if month in monthly_returns:  # Check if the month exists in the returns
                    monthly_aggregated_returns[months[month-1]].extend(monthly_returns[month])

        # Calculate average returns
        final_average_returns = {month: np.mean(returns) if returns else None for month, returns in monthly_aggregated_returns.items()}
        return self.plot_average_monthly_returns(final_average_returns)

    def plot_average_monthly_returns(self, average_returns):
        """Plot the average monthly returns using data retrieved from get_average_monthly_returns."""
        values = list(average_returns.values())
        labels = list(average_returns.keys())
        x = np.arange(len(values))

        fig, ax = plt.subplots()
        # Add a label here to be used in the legend
        bars = ax.bar(x, values, alpha=0.7, label='Monthly Average Return')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Return (%)')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        date = self.return_calculator.earliest_data_year
        ticker = self.return_calculator.tickers
        title = f"{ticker[0].strip()} Average Monthly Returns since {date}"
        ax.set_title(title)
        ax.set_ylabel('Average Return (%)')
        plt.legend()
        plt.tight_layout()
        file_path: str = f'Monthly Charts/'
        plt.savefig(file_path + title + ".png")
        plt.show()

    def get_average_semi_monthly_returns(self):
        """Retrieve and plot average semi-monthly returns for all tickers."""
        average_returns = None
        # Predefine all possible periods for consistent ordering
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        periods = [f"{month} H{i}" for month in months for i in [1, 2]]
        aggregated_returns = pd.DataFrame(index=periods)  # Initialize DataFrame with predefined periods as index

        for ticker in self.return_calculator.tickers:
            semi_monthly_returns = self.return_calculator.calculate_semi_monthly_returns(ticker)
            if semi_monthly_returns.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Aggregate returns into the DataFrame
            aggregated_returns = pd.concat([aggregated_returns, semi_monthly_returns], axis=1)

        if aggregated_returns.empty:
            print("No data available for any tickers.")
            return None
        # Calculate the mean across all tickers for each semi-monthly period
        average_returns = aggregated_returns.mean(axis=1)

        return self.plot_average_semi_monthly_returns(average_returns)

    def plot_average_semi_monthly_returns(self, average_returns):
        periods = average_returns.index.tolist()  # Assuming average_returns is a pandas Series
        values = average_returns.values
        x = np.arange(len(periods))

        fig, ax = plt.subplots()
        # Include a label to be referenced in the legend
        bars = ax.bar(x, values, alpha=0.7, label='Semi-Monthly Average Return')
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=90)
        ax.set_xlabel('Semi-Monthly Periods')
        ax.set_ylabel('Average Return (%)')
        date = self.return_calculator.earliest_data_year
        ticker = self.return_calculator.tickers
        title = str(ticker[0].strip()) + " Average Semi-Monthly Returns since " + str(date)
        ax.set_title(title)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.legend()
        plt.tight_layout()
        file_path: str = f'Semi-Monthly Charts/'
        plt.savefig(file_path + title +".png")
        plt.show()


    def get_average_monthly_volatility(self):
        """Retrieve average monthly volatility for all tickers and prepare for plotting."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_aggregated_returns = {month: [] for month in months}

        for ticker in self.return_calculator.tickers:
            monthly_returns = self.return_calculator.calculate_monthly_returns(ticker)
            if monthly_returns.empty:  # Correct way to check if a DataFrame or Series is empty
                print(f"No data available for ticker {ticker}")
                continue

            for month in range(1, 13):  # Month indices from 1 to 12
                if month in monthly_returns:  # Check if the month exists in the returns
                    monthly_aggregated_returns[months[month-1]].extend(monthly_returns[month])

        # Calculate average returns
        final_average_returns = {month: np.std(returns) if returns else None for month, returns in monthly_aggregated_returns.items()}
        return self.plot_average_monthly_volatility(final_average_returns)

    def plot_average_monthly_volatility(self, average_returns):
        """Plot the average monthly returns using data retrieved from get_average_monthly_returns."""
        values = list(average_returns.values())
        labels = list(average_returns.keys())
        x = np.arange(len(values))

        fig, ax = plt.subplots()
        # Add a label here to be used in the legend
        bars = ax.bar(x, values, alpha=0.7, label='Monthly Average Volatility')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Volatility (%)')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        date = self.return_calculator.earliest_data_year
        ticker = self.return_calculator.tickers
        title = f"{ticker[0].strip()} Average Monthly Volatility since {date}"
        ax.set_title(title)
        ax.set_ylabel('Average Volatility (%)')
        plt.legend()
        plt.tight_layout()
        file_path: str = f'Monthly Charts/'
        # plt.savefig(file_path + title + ".png")
        plt.show()

    def plot_price_sma_difference(self, window=30):
        """
        Plot the difference between the index prices and their exponential moving averages (EMA) for all tickers in the ReturnCalculator.
        EMA is calculated using the entire dataset, but only data between 2021-06-28 and 2023-06-28 is displayed.

        Parameters:
        - window: integer, the window size for the moving average (default is 30 days)
        """
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate the exponential moving average over the entire dataset
            data['EMA'] = data['Adj Close'].ewm(span=window, adjust=False).mean()

            # Calculate the difference between price and EMA
            data['Price-EMA Difference'] = data['Adj Close'] - data['EMA']

            # Filter the data between June 28, 2021, and June 28, 2023, for plotting
            filtered_data = data['2021-06-28':'2023-06-28']

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plotting the price and EMA for the specified period
            ax.plot(filtered_data['Adj Close'], color='limegreen', label='Index Price')
            ax.plot(filtered_data['EMA'], color='mediumblue', label='Exponential Moving Average', linewidth=0.8)

            ax.set_ylabel('Price / EMA')
            ax.set_title(f'Index Price and EMA Comparison for {ticker}')

            # Plotting the difference on a secondary axis
            ax2 = ax.twinx()
            ax2.plot(filtered_data['Price-EMA Difference'], color='blue', label='Price-EMA Difference', alpha=0.7)
            ax2.set_ylabel('Price-EMA Difference')
            ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

            # Legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')

            plt.show()

