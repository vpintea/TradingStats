# https://d-nb.info/1174250364/34#page=24&zoom=100,0,0


import return_calculator as ReturnCalculator
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from entropy import get_entropy
import statsmodels.api as sm

# Set the backend to a non-inline backend that supports interactive windows
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your preference
# import plotly.graph_objects as go

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

    def plot_return_momentum(self):
        """Plot the ReturnMomentum indicator against the index price."""
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate Returns and ReturnMomentum
            data['Returns'] = data['Adj Close'].pct_change()
            data['ReturnMomentum'] = data['Returns'] - data['Returns'].shift(1)

            # Filter data for the desired period
            filtered_data = data.loc['2021-06-28':'2023-06-28']

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Index price in green
            ax.plot(filtered_data.index, filtered_data['Adj Close'], label='Index Price', color='limegreen')

            # ReturnMomentum in blue
            ax2 = ax.twinx()  # Create a second y-axis
            ax2.plot(filtered_data.index, filtered_data['ReturnMomentum'], label='ReturnMomentum', color='blue',
                     alpha=0.7)

            # Setting labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Price')
            ax2.set_ylabel('ReturnMomentum')

            # Legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            ax.set_title(f'Return Momentum for {ticker}')
            plt.show()

    def plot_return_acceleration(self):
        """Plot the ReturnAcceleration indicator against the index price."""
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate Returns and ReturnAcceleration
            data['Returns'] = data['Adj Close'].pct_change()
            data['ReturnMomentum'] = data['Returns'] - data['Returns'].shift(1)
            data['ReturnAcceleration'] = data['ReturnMomentum'] - data['ReturnMomentum'].shift(1)

            # Filter data for the desired period
            filtered_data = data.loc['2021-06-28':'2023-06-28']

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 7))

            # Index price in green
            ax.plot(filtered_data.index, filtered_data['Adj Close'], label='Index Price', color='limegreen', alpha=0.7)

            # ReturnAcceleration in blue on a secondary axis
            ax2 = ax.twinx()  # Create a second y-axis
            ax2.plot(filtered_data.index, filtered_data['ReturnAcceleration'], label='ReturnAcceleration',
                     color='mediumblue', alpha=0.7)

            # Setting labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Price')
            ax2.set_ylabel('ReturnAcceleration')

            # Legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            ax.set_title(f'Return Acceleration for {ticker}')
            plt.show()

    def plot_week_price_momentum(self):
        """Plot the natural logarithm of WeekPriceMomentum against the index price."""
        start_date = '2021-06-28'
        end_date = '2023-06-28'

        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate the ratio of the current price to the price a week ago
            data['Price_Ratio'] = data['Adj Close'] / data['Adj Close'].shift(5)

            # Calculate the natural logarithm of the price ratio and multiply by 100 to scale it similar to percentage
            data['LnWeekPriceMomentum'] = np.log(data['Price_Ratio'])

            # Filter data for the desired period
            filtered_data = data.loc[start_date:end_date]

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            # Index price in green
            ax.plot(filtered_data.index, filtered_data['Adj Close'], label='Index Price', color='limegreen', alpha=0.7)

            # LnWeekPriceMomentum in blue on a secondary axis
            ax2 = ax.twinx()  # Create a second y-axis
            ax2.plot(filtered_data.index, filtered_data['LnWeekPriceMomentum'], label='Ln WeekPriceMomentum',
                     color='mediumblue', alpha=0.6)

            # Adding a horizontal line at zero for LnWeekPriceMomentum
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)

            # Setting labels and titles
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Price')
            ax2.set_ylabel('Ln WeekPriceMomentum (%)')

            # Legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            ax.set_title(f'Logarithmic Weekly Price Momentum for {ticker}')
            plt.show()

    def plot_month_price_momentum(self):
        """Plot the MonthPriceMomentum and index price."""
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate MonthPriceMomentum
            data['MonthPriceMomentum'] = data['Adj Close'].pct_change(
                periods=20) * 100  # Assuming ~20 trading days in a month

            # Filter data for a specific time frame
            filtered_data = data['2021-06-28':'2023-06-28']

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 7))

            # Index price in green
            ax.plot(filtered_data.index, filtered_data['Adj Close'], label='Index Price', color='limegreen', alpha=0.7)

            # MonthPriceMomentum in blue on a secondary axis
            ax2 = ax.twinx()
            ax2.plot(filtered_data.index, filtered_data['MonthPriceMomentum'], label='MonthPriceMomentum',
                     color='mediumblue', alpha=0.7)

            # Adding a horizontal line at zero for MonthPriceMomentum
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)

            # Setting labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Price')
            ax2.set_ylabel('MonthPriceMomentum (%)')

            # Legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            ax.set_title(f'Monthly Price Momentum for {ticker}')
            plt.show()

    def plot_volume_velocity(self):
        """Plot the VolumeVelocity alongside the trading volume."""
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

            # Calculate VolumeVelocity as the percentage change in volume from the previous day
            data['VolumeVelocity'] = data['Volume'].pct_change() * 100

            # Filter data for the desired date range
            filtered_data = data['2021-06-28':'2023-06-28']

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 7))

            # Volume in blue
            ax.bar(filtered_data.index, filtered_data['Volume'], label='Volume', color='blue', alpha=0.5)

            # VolumeVelocity in red on a secondary axis
            ax2 = ax.twinx()
            ax2.plot(filtered_data.index, filtered_data['VolumeVelocity'], label='VolumeVelocity', color='red',
                     alpha=0.7)

            # Adding a horizontal line at zero for VolumeVelocity
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)

            # Setting labels
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            ax2.set_ylabel('VolumeVelocity (%)')

            # Legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            ax.set_title(f'Volume and Volume Velocity for {ticker}')
            plt.show()

    def _plot_skewness_entropy_and_returns(self, ticker: str):
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue
        # Calculate daily returns
        daily_returns = self.return_calculator.calculate_daily_returns(ticker)
        entropy_df = get_entropy()
        # print(entropy_df)
        # Merge dataframes to ensure proper alignment of dates
        merged_df = pd.merge(entropy_df, daily_returns, left_on='Date', right_index=True, how='inner')
        # print(merged_df)
        # Plotting the results
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Tsallis Entropy
        ax1.plot(merged_df['Date'], merged_df['Entropy'], color='blue', marker='o', markersize=2, label='Tsallis Entropy')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Entropy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for the skewness
        ax2 = ax1.twinx()
        ax2.plot(merged_df['Date'], merged_df['Skewness'], color='red', marker='x', markersize=2, label='Skewness')
        ax2.set_ylabel('Skewness', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Create a third y-axis for the daily returns
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis to the right
        ax3.plot(merged_df['Date'], merged_df['Adj Close'], color='green', marker='^', markersize=2, label='SPX Price')
        ax3.set_ylabel('SPX Price', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        # Title and grid
        plt.title(f'Tsallis Entropy, Skewness, and SPX Index for {ticker} Over Time')
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def plot_skewness_entropy_and_returns(self, ticker: str):
        # Fetch the data for the ticker
        data = self.return_calculator.financial_data[ticker].get_data()
        if data.empty:
            print(f"No data available for ticker {ticker}")
            return

        # Calculate daily returns
        daily_returns = self.return_calculator.calculate_daily_returns(ticker)

        # Get entropy values
        entropy_df = get_entropy()

        # Merge dataframes to ensure proper alignment of dates
        merged_df = pd.merge(entropy_df, daily_returns, left_on='Date', right_index=True, how='inner')

        # Create the figure using Plotly
        fig = go.Figure()

        # Entropy (blue)
        fig.add_trace(go.Scatter(
            x=merged_df['Date'],
            y=merged_df['Entropy'],
            mode='lines+markers',
            name='Entropy',
            marker=dict(color='blue', size=3),
            line=dict(color='blue'),
            yaxis='y1',
            hovertemplate='Entropy: %{y:.4f}<extra></extra>'
        ))

        # SPX Price (green) on the primary y-axis
        fig.add_trace(go.Scatter(
            x=merged_df['Date'],
            y=merged_df['Adj Close'],
            mode='lines+markers',
            name='SPX Price',
            marker=dict(color='green', symbol='triangle-up', size=4),
            line=dict(color='green'),
            yaxis='y2',
            hovertemplate='SPX Price: %{y:.2f}<extra></extra>'
        ))

        # Skewness (red)
        fig.add_trace(go.Scatter(
            x=merged_df['Date'],
            y=merged_df['Skewness'],
            mode='lines+markers',
            name='Skewness',
            marker=dict(color='red', symbol='x', size=4),
            line=dict(color='red'),
            yaxis='y3',
            hovertemplate='Skewness: %{y:.4f}<extra></extra>'
        ))

        # Update layout for multiple y-axes
        fig.update_layout(
            title=f'Entropy, Skewness, and SPX Index for {ticker} Over Time',
            xaxis=dict(title='Date', tickformat='%Y-%m-%d'),
            yaxis=dict(
                title='Entropy',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                showgrid=False,
            ),
            yaxis2=dict(
                title='SPX Price',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                overlaying='y',  # Overlays SPX Price on the same axis
                side='right',
                showgrid=False,
                automargin=True,  # Ensure the y-axis adjusts based on data range when zoomed
                range=[merged_df['Adj Close'].min(), merged_df['Adj Close'].max()]  # SPX price range
            ),
            yaxis3=dict(
                title='Skewness',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',  # Overlays Skewness on the same axis
                side='right',
                position=0.95,  # Moves Skewness to the far right
                showgrid=False,
                automargin=True
            ),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            template='plotly_white'
        )

        # Show the plot in a browser
        fig.show()

    def calculate_normalized_momentum_indicators(self, df, N=14):
        """
        Calculate and normalize multiple momentum indicators with a midpoint of 0.
        """
        # KMID: ($close - $open) / $open
        df['KMID'] = (df['Close'] - df['Open']) / df['Open']

        # KLEN: (($high - $low) / $open)
        df['KLEN'] = (df['High'] - df['Low']) / df['Open']

        # KUP: (($high - max($open, $close)) / $open)
        df['KUP'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open']

        # KLOW: ((min($open, $close) - $low) / $open)
        df['KLOW'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open']

        # ROC: (REF(($close, N) / $close))
        df['ROC'] = df['Close'].pct_change(N)

        # STD: Standard deviation of $close over the last N periods
        df['STD'] = df['Close'].rolling(window=N).std() / df['Close']

        # RSV: ($close - MIN($low, N)) / (MAX($high, N) - MIN($low, N))
        df['RSV'] = (df['Close'] - df['Low'].rolling(window=N).min()) / (
                df['High'].rolling(window=N).max() - df['Low'].rolling(window=N).min()
        )

        # Normalize each indicator by subtracting its mean to center around 0
        for column in ['KMID', 'KLEN', 'KUP', 'KLOW', 'ROC', 'STD', 'RSV']:
            df[column] = df[column] - df[column].mean()

        # Calculate average of all normalized indicators
        df['Momentum_Average'] = df[['KMID', 'KLEN', 'KUP', 'KLOW', 'ROC', 'STD', 'RSV']].mean(axis=1)

        return df

    def plot_normalized_momentum(self, ticker: str, N=7):
        """
        Plot the normalized average momentum indicators for a given ticker using matplotlib.
        """
        for ticker in self.return_calculator.tickers:
            data = self.return_calculator.financial_data[ticker].get_data()
            if data.empty:
                print(f"No data available for ticker {ticker}")
                continue

        # Calculate normalized momentum indicators and their average
        momentum_df = self.calculate_normalized_momentum_indicators(data, N=N)

        # Plot using matplotlib
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot normalized Average Momentum
        ax1.plot(momentum_df.index, momentum_df['Momentum_Average'], label='Normalized Avg Momentum', color='blue', marker='o', markersize=3)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Avg Momentum', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for SPX Price
        ax2 = ax1.twinx()
        ax2.plot(momentum_df.index, momentum_df['Adj Close'], label='SPX Price', color='green', marker='^', markersize=3)
        ax2.set_ylabel('SPX Price', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Title and layout
        plt.title(f'Normalized Average Momentum and SPX Price for {ticker} Over Time')
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    # data = data['2021-06-28':'2023-06-28']

