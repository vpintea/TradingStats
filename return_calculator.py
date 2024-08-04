import pandas as pd
from financial_data import FinancialData
from typing import List, Dict
import numpy as np


class ReturnCalculator:
    def __init__(self, tickers: List[str], start_date: str = '1960-01-01') -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.financial_data = {ticker: FinancialData(ticker, start_date) for ticker in tickers}
        self.earliest_data_year = None

    def calculate_semi_monthly_returns(self, ticker: str):
        data = self.financial_data[ticker].get_data()
        if data is None or data.empty:
            return {}

        data['Period'] = data.index.to_series().apply(lambda x: f"{x.strftime('%b')} {'H1' if x.day <= 15 else 'H2'}")
        data['Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)
        self.earliest_data_year = data.index.min().year if data.index.min().year > int(
            self.start_date.split("-")[0]) else int(self.start_date.split("-")[0])
        # Group by 'Period' and calculate product of returns
        grouped = data.groupby(['Period', data.index.year])
        semi_monthly_returns = grouped['Return'].apply(lambda x: (1 + x).prod() - 1) * 100

        return semi_monthly_returns.unstack()

    def calculate_average_quarterly_returns_cycle(self, ticker: str) -> Dict[str, Dict[str, list]]:
        data = self.financial_data[ticker].get_data()
        if data is None or data.empty:
            return {}

        data['Quarter'] = data.index.to_period('Q')
        data['Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)

        quarterly_returns = data.groupby('Quarter')['Return'].apply(lambda x: (1 + x).prod() - 1) * 100

        # Dynamic calculation of the election years based on data's date range
        current_year = pd.Timestamp.now().year
        latest_election_year = current_year if current_year % 4 == 0 else current_year - (current_year % 4)

        # Calculate the start year for cycle calculations, which is the nearest past election year from the data's earliest date
        self.earliest_data_year = data.index.min().year if data.index.min().year > int(self.start_date.split("-")[0]) else int(self.start_date.split("-")[0])
        start_cycle_year = self.earliest_data_year - (self.earliest_data_year % 4) if self.earliest_data_year % 4 != 0 else self.earliest_data_year

        # Initialize dictionary for each cycle
        average_returns = {f'n-{(latest_election_year - year) // 4}': {q: [] for q in ['Q1', 'Q2', 'Q3', 'Q4']}
                          for year in range(start_cycle_year, latest_election_year + 1, 4)}
        years = data.index.year.unique()
        election_years = [year for year in years if year % 4 == 0]

        # Populate the dictionary with quarterly returns
        for period, return_value in quarterly_returns.items():
            year = period.year
            quarter = period.quarter
            if year > election_years[0]:
                election_years.pop(0)
            cycle_index = (election_years[0] - year) % 4
            cycle_key = f'n-{cycle_index}'
            average_returns[cycle_key][f'Q{quarter}'].append(return_value)

        # Print all cycles to debug
        # for key, value in average_returns.items():
        #     print(f"{key}: {value}")
        return average_returns

    def calculate_monthly_returns(self, ticker):
        data = self.financial_data[ticker].get_data()
        if data is None or data.empty:
            return {}

        data['Month'] = data.index.to_period('M')
        data['Return'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)
        self.earliest_data_year = data.index.min().year if data.index.min().year > int(self.start_date.split("-")[0]) else int(self.start_date.split("-")[0])
        # Group by month and calculate average of returns
        monthly_returns = data.groupby('Month')['Return'].apply(lambda x: (1 + x).prod() - 1) * 100
        return monthly_returns.groupby(monthly_returns.index.month).agg(list)  # Group by month number and aggregate as list