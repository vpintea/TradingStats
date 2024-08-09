# Download stock data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']


# Calculate simple returns
def calculate_returns(data):
    return data.pct_change().dropna()


# Create a 3D plot of stocks colored by sector (this is a simplified example)
def plot_stocks_3d(data_dict, sector_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = {'tech': 'red', 'finance': 'blue', 'healthcare': 'green'}

    for ticker, data in data_dict.items():
        ax.scatter(data['X'], data['Y'], data['Z'], color=colors[sector_dict[ticker]], label=ticker)

    ax.set_xlabel('Attribute X')
    ax.set_ylabel('Attribute Y')
    ax.set_zlabel('Attribute Z')
    ax.legend()
    plt.show()


# Main function to run the analysis
def main():
    tickers = ['AAPL', 'JPM', 'PFE']  # Example tickers
    sectors = {'AAPL': 'tech', 'JPM': 'finance', 'PFE': 'healthcare'}
    start_date = '2020-01-01'
    end_date = '2021-01-01'

    stock_data = {}
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        returns = calculate_returns(data)
        # For simplicity, just random attributes for X, Y, Z
        stock_data[ticker] = {
            'X': np.random.normal(loc=0, scale=1, size=len(returns)),
            'Y': np.random.normal(loc=0, scale=1, size=len(returns)),
            'Z': returns
        }

    plot_stocks_3d(stock_data, sectors)

if __name__ == "__main__":
    main()