import pandas as pd
import yfinance as yf
import os


class FinancialData:
    def __init__(self, ticker: str, start_date: str = '1960-01-01') -> None:
        self.ticker: str = ticker
        self.start_date: str = start_date
        self.data: pd.DataFrame = None
        self.file_path: str = f'historical_data/{self.ticker}.csv'

    def download_data(self) -> pd.DataFrame:
        """Download historical data for the given ticker."""
        data = yf.download(self.ticker, start=self.start_date)
        # data = data[['Adj Close', 'Volume']]  # Keep only 'Adj Close' column
        self.data = data.dropna()
        return self.data

    def save_data(self) -> None:
        """Save the downloaded data to a CSV file."""
        if self.data is not None:
            self.data.to_csv(self.file_path)

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file if it exists."""
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        return self.data

    def get_data(self) -> pd.DataFrame:
        """Returns the downloaded data."""
        if self.data is None:
            self.load_data()
        if self.data is None or not self.is_data_up_to_date():
            self.download_data()
            self.save_data()
        return self.data

    def is_data_up_to_date(self) -> bool:
        """Check if the data is up-to-date."""
        if self.data is not None:
            last_date = self.data.index[-1]
            today = pd.Timestamp.today()
            if today.weekday() >= 5:  # if today is Saturday (5) or Sunday (6), check for last Friday
                today -= pd.Timedelta(days=today.weekday() - 4)
            return last_date >= today
        return False
