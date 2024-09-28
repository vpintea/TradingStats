# http://www.rcea.org/RePEc/pdf/wp15-14.pdf - research paper on 2 entropies
# https://www.optionsdx.com/shop/ - 2023 and prior data downloaded from here
# https://pages.stern.nyu.edu/~dbackus/GE_asset_pricing/disasters/Bates%20crash%20JF%2091.PDF - initial Bates paper that used quarterly options (1-4 mo out only)
# https://elsevier-ssrn-document-store-prod.s3.amazonaws.com/07/01/25/ssrn_id959547_code741880.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHIaCXVzLWVhc3QtMSJHMEUCIQDBk4CivXUsMuwDq0%2FYazNQB8rtYG2kW%2FVjQ0Q9cwdXTgIgD9UEhzMvm5xT5m8IQIR2cIGOFwD3%2BO44k3mf6KfxVw4qvQUIexAEGgwzMDg0NzUzMDEyNTciDJYabTyDi9zLjewD1CqaBd8vdyrzZ38WmaxYRHqiXsPZC1hKTxcHdf3owBsMbw37LNLcGrF5eOX9weApRU7ehTpUW4nyupsUqiCzkRngVXoRcsDpaz9g%2Fc2ydY4BwvqUAZ38nW7xWMGoxFwbvwEXagU6voJ439%2BTQHOVPMh0y9G6VsqCXOu9ETmzxsJKp%2B8Jx9QjvDVq55MvUVpryeeMeS%2FqzKsodVp2nKaxbqbeOtJALA0sABJwkgQiHw8CX73hTa6B1LhlIaHrwUCDeb3DKKngAZrvs6Rvax7jE%2FHntk0NMgMltItZVXaHZTLRc68LfjwODNZO%2Bo34g4x6sX7MgLWFFFJeZ%2F8m%2FhV3vBS8a92MEUooapsOWAFVFy0K0zk1IvatMsiC%2BPtPw0YkgjBiZkrxR0UeiLkrv2qGvMIdLJLMaMExUpcAT6cqSnQwXA7zhJYIw%2FDEthWMvs9ydXwMqfw0UgPkhKaszmKWuT0%2BGrmqYBIgUUQK%2BJqMpBhAViy45bJ1GSE2lhc11I6Ssq%2BXSv9JkeQfG6CMIdz1bgwCYmH2eXCXcWthDEPe4LTbinzNWl%2FEUM5ywepua8jLXsXQAoijxgS4CxHeJfAb9utvp5Zj%2BhUO%2BhAwwzryo6rQgLfKEb4uqlV0vr6FRL8CJ1X%2BGCK3NPEwcJgfdRzIdMaNgads26Szr471tgm7CSyZhNDhCyLV6QMHAs00LkthEIEN9M8ysi2%2B0kTrZDLX1zu7jHbi3GPBzG7ruyC%2FawuEGp3vuc2k4H0WacSoASwGFethu15gO7ul9EdNNPFACqPArhwa0XfYB7Y89YKxKemGAM5Ag7ucFFLh7czMA0U%2FtMmnpLuB42GaP5ZxGysKH2tLo9cxq1BKUfxqaMtcLXfz%2Ft7DlfGgVUNlSLCJ5jDB1pi2BjqxAVzvN%2FU0msnO7%2BuSWI0VfH%2F8Xm5cSL7ajYBYIiq%2FWMpJBYAiGJvwPPL791Xc%2BtTQTxd9n2mFnwZS%2FTZfUtwyF9i6p%2FWw9fnZW%2Ba0c2yckqO32zx4nOnVKAKJJYmUP76zTVqHZqtLNbvpzdDMzzhvbf3c4JD6c3c%2FS4akVWVnmsmH%2B5wWK275s38OJ%2BQ4U%2BjizFy6V%2FMI4%2F1vev3wpqG1cxguxi00FxVUBBXPN11hQTt0xw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240821T184814Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWER6KXSCFA%2F20240821%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=bc7f0a76da27f7c7166e234ba9fb71a99acd485caaa6d5e81595ff3d58ef3f20
# daily options data for download: https://www.cboe.com/delayed_quotes/spx/quote_table

# Tsalllis entropy best for predicting 1-day crashes like 1987
# Approximate entropy best for predicting longer-slower-duration crashes like 2008

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import load_options_data_from_db
from load_daily_csv_to_db import load_daily_data_to_db

pd.set_option('display.max_columns', None)
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your preference

def get_entropy():
    start_time = time.time()  # Start timing
    # option_data_sample = load_and_sort_cleaned_option_data(file_name="cleaned_option_data_final.csv", start_date=start_date, end_date=end_date)
    start_date = "2019-05-11"
    end_date = "2024-09-18"

    load_db_start = time.time()
    load_daily_data_to_db()
    print(f"Loading daily data to DB took {time.time() - load_db_start:.2f} seconds")

    load_data_start = time.time()
    option_data_sample = load_options_data_from_db(start_date)
    print(f"Loading options data from DB took {time.time() - load_data_start:.2f} seconds")

    clean_data_start = time.time()
    option_data_sample.dropna(subset=['quote_date', 'expire_date', 'c_bid', 'c_ask', 'p_bid', 'p_ask'], inplace=True)
    # Calculate the average price for call and put options
    option_data_sample['C_Avg'] = (option_data_sample['c_bid'] + option_data_sample['c_ask']) / 2
    option_data_sample['P_Avg'] = (option_data_sample['p_bid'] + option_data_sample['p_ask']) / 2
    # Ensure we're only looking at rows where C_Avg and P_Avg are not NaN
    option_data_sample.dropna(subset=['C_Avg', 'P_Avg'], inplace=True)
    print(f"Data cleaning and preprocessing took {time.time() - clean_data_start:.2f} seconds")

    filter_data_start = time.time()
    # Remove rows where C_Avg or P_Avg is 0
    option_data_sample = option_data_sample[(option_data_sample['C_Avg'] != 0) & (option_data_sample['P_Avg'] != 0)]
    # option_data_sample = option_data_sample[(option_data_sample['C_Avg'] != 0)]
    option_data_sample['Total Volume'] = (option_data_sample['c_volume'] + option_data_sample['p_volume'])
    # Filter for options that expire in March, June, September, or December
    # option_data_sample = option_data_sample[option_data_sample['expire_date'].dt.month.isin([3, 6, 9, 12])]
    # Filter the DataFrame where DTE is greater than 28 and less than 200
    filtered_data = option_data_sample[(option_data_sample['dte'] > 28) & (option_data_sample['dte'] < 118)]
    # filtered_data = filtered_data[filtered_data['Total Volume'] > 0]
    # filtered_data = filtered_data[(filtered_data['quote_date'] > "2024-07-12") & (filtered_data['quote_date'] < "2024-07-20")]
    # After filtering, you can group by 'QUOTE_DATE' or any other operation you wish to perform
    # grouped = filtered_data.groupby(['quote_date'])
    print(f"Filtering data took {time.time() - filter_data_start:.2f} seconds")

    skewness_calc_start = time.time()

    def calculate_skewness_for_nearest_expiry(df):
        """
        Calculate skewness premium using the nearest expiry for each quote_date.

        :param df: DataFrame containing options data
        :return: DataFrame with skewness premium for each quote_date
        """

        # Step 1: Filter for options expiring on or after the 15th of the month
        df['expire_day'] = df['expire_date'].dt.day
        df = df[df['expire_day'] >= 15]  # Focus on expiries on or after the 15th of the month

        df = df.sort_values(by=['quote_date', 'expire_date'])

        # Group by 'quote_date'
        grouped = df.groupby('quote_date')

        def skewness_for_group(group):
            # Identify the nearest expiry date
            nearest_expiry = group['expire_date'].min()

            # Filter options for the nearest expiry date
            nearest_expiry_options = group[group['expire_date'] == nearest_expiry]

            # Sort options by strike price
            nearest_expiry_options_sorted = nearest_expiry_options.sort_values(by='strike')

            # Find the deepest OTM call and put
            deepest_otm_call = nearest_expiry_options_sorted.iloc[-1]  # Last row (highest strike call)
            deepest_otm_put = nearest_expiry_options_sorted.iloc[0]  # First row (lowest strike put)

            # Ensure both have valid data
            if pd.isna(deepest_otm_call['C_Avg']) or pd.isna(deepest_otm_put['P_Avg']):
                print("Skipping calculation: Invalid data for max call or min put.")
                return None

            call_avg = deepest_otm_call['C_Avg']
            put_avg = deepest_otm_put['P_Avg']

            # Skip calculation if call_avg is 0 to avoid division by zero
            if call_avg == 0:
                return None

            # Calculate skewness premium
            skewness_premium = (put_avg / call_avg) - 1
            return skewness_premium

        # Apply skewness calculation for each quote_date
        skewness_data = grouped.apply(skewness_for_group).reset_index()
        skewness_data.columns = ['quote_date', 'skewness_premium']

        return skewness_data
    def calculate_skewness_for_group(df):
        group = df

        # Check if there are rows left after filtering
        if group.empty:
            print("No data available after filtering by QUOTE_DATE")
            return None

        # Sort the group by strike prices
        group_sorted = group.sort_values(by='strike')

        # Find the farthest OTM call and put
        max_call = group_sorted[group_sorted['C_Avg'] > 0].iloc[-1] if not group_sorted[
            group_sorted['C_Avg'] > 0].empty else None
        min_put = group_sorted[group_sorted['P_Avg'] > 0].iloc[0] if not group_sorted[
            group_sorted['P_Avg'] > 0].empty else None

        # Ensure that both call and put have valid data
        if max_call is None or min_put is None:
            print("Skipping calculation: Invalid data for max call or min put.")
            return None

        call_avg = max_call['C_Avg']
        put_avg = min_put['P_Avg']

        # Skip calculation if call_avg is 0 to avoid division by zero
        if call_avg == 0:
            print(f"Skipping calculation: call_avg is 0 for group with QUOTE_DATE = {group.iloc[0]['quote_date']}")
            return None

        # Calculate skewness premium
        skewness_premium = (put_avg / call_avg) - 1

        return skewness_premium

    def _calculate_skewness_for_group(df):
        """
        Calculates the skewness premium for each group by finding the farthest OTM call and put
        that are equidistant from the underlying price. If no exact match is found, it will select
        the closest matching pair.

        Args:
        df (pd.DataFrame): The DataFrame containing option data for a specific QUOTE_DATE and EXPIRE_DATE.

        Returns:
        float or None: The calculated skewness premium, or None if no valid matches are found.
        """
        group = df

        if group.empty:
            print("No data available after filtering by QUOTE_DATE and EXPIRE_DATE")
            return None

        # Sort the group by strike prices
        group_sorted = group.sort_values(by='strike')

        # Calculate OTM percentages for calls and puts
        group_sorted['Call_OTM_Perc'] = (group_sorted['strike'] - group_sorted['underlying_last']) / group_sorted[
            'underlying_last'] * 100
        group_sorted['Put_OTM_Perc'] = (group_sorted['underlying_last'] - group_sorted['strike']) / group_sorted[
            'underlying_last'] * 100

        # Filter for valid OTM calls (positive OTM percentage) and puts (positive OTM percentage)
        call_candidates = group_sorted[group_sorted['Call_OTM_Perc'] > 0]
        put_candidates = group_sorted[group_sorted['Put_OTM_Perc'] > 0]

        if call_candidates.empty or put_candidates.empty:
            # print(f"No valid OTM calls or puts found for QUOTE_DATE = {group.iloc[0]['quote_date']}")
            return None

        # Initialize variables to store the best matching call/put pair
        best_call = None
        best_put = None
        smallest_diff = float('inf')

        # Iterate through the farthest OTM calls and find the best matching put
        for _, call in call_candidates.iterrows():
            call_otm = call['Call_OTM_Perc']
            matching_put = put_candidates.iloc[(put_candidates['Put_OTM_Perc'] - call_otm).abs().argsort()[
                                               :1]]  # Find the closest put by OTM percentage

            if not matching_put.empty:
                put_otm = matching_put.iloc[0]['Put_OTM_Perc']
                diff = abs(call_otm - put_otm)

                if diff < smallest_diff:  # Update if we find a closer match
                    smallest_diff = diff
                    best_call = call
                    best_put = matching_put.iloc[0]

        # Check if valid best_call and best_put were found
        if best_call is None or best_put is None:
            print(f"No valid matching OTM calls or puts found for QUOTE_DATE = {group.iloc[0]['quote_date']}")
            return None

        # Get the average prices for the best matching call and put
        call_avg = best_call['C_Avg']
        put_avg = best_put['P_Avg']

        # Skip calculation if call_avg is 0 to avoid division by zero
        if call_avg == 0:
            # print(f"Skipping calculation: call_avg is 0 for QUOTE_DATE = {group.iloc[0]['quote_date']}")
            return None

        # Calculate skewness premium
        skewness_premium = (put_avg / call_avg) - 1

        return skewness_premium

    # Apply the function to each group
    # skewness_data = grouped.apply(calculate_skewness_for_group).reset_index()


    def calculate_farthest_otm_skewness(group):
        # Sort the group by expiry and strike to find the farthest OTM options for one expiry
        group_sorted = group.sort_values(by=['expire_date', 'strike'])

        # Find the expiry with the farthest OTM options
        best_expiry = None
        best_skewness = None

        # Iterate over each expiry to find the farthest OTM call and put
        for expiry, expiry_group in group_sorted.groupby('expire_date'):
            call_candidates = expiry_group[expiry_group['C_Avg'] > 0]
            put_candidates = expiry_group[expiry_group['P_Avg'] > 0]

            if call_candidates.empty or put_candidates.empty:
                continue

            # Get the farthest OTM call and put for this expiry
            max_call = call_candidates.iloc[-1]
            min_put = put_candidates.iloc[0]

            call_avg = max_call['C_Avg']
            put_avg = min_put['P_Avg']

            # Skip calculation if call_avg is 0 to avoid division by zero
            if call_avg == 0:
                continue

            # Calculate skewness premium
            skewness_premium = (put_avg / call_avg) - 1

            # Track the best expiry based on the furthest OTM option
            if best_skewness is None or abs(skewness_premium) > abs(best_skewness):
                best_expiry = expiry
                best_skewness = skewness_premium

        return best_skewness

    def calculate_skewness_for_multiple_expiries(df):
        """
        Calculate the average skewness premium for multiple expiries for each quote_date.

        :param df: DataFrame containing options data
        :return: DataFrame with average skewness premium for each quote_date
        """

        # Sort the data by quote_date and expire_date to ensure proper ordering
        df = df.sort_values(by=['quote_date', 'expire_date'])

        # Group the data by quote_date and expire_date
        grouped = df.groupby(['quote_date', 'expire_date'])

        def skewness_for_group(group):
            """
            Calculate skewness premium for a specific quote_date and expiry.
            """
            # Sort options by strike price
            group_sorted = group.sort_values(by='strike')

            # Find the deepest OTM call and put (highest strike call and lowest strike put)
            deepest_otm_call = group_sorted.iloc[-1]  # Last row (highest strike call)
            deepest_otm_put = group_sorted.iloc[0]  # First row (lowest strike put)

            # Ensure both have valid data
            if pd.isna(deepest_otm_call['C_Avg']) or pd.isna(deepest_otm_put['P_Avg']):
                return None

            call_avg = deepest_otm_call['C_Avg']
            put_avg = deepest_otm_put['P_Avg']

            # Skip calculation if call_avg is 0 to avoid division by zero
            if call_avg == 0:
                return None

            # Calculate skewness premium
            skewness_premium = (put_avg / call_avg) - 1
            return skewness_premium

        # Apply skewness calculation to each group (quote_date, expire_date)
        skewness_data = grouped.apply(skewness_for_group).reset_index(name='skewness_premium')

        # Now, group by 'quote_date' and calculate the average skewness for each quote_date
        avg_skewness_data = skewness_data.groupby('quote_date').agg({'skewness_premium': 'median'}).reset_index()

        avg_skewness_data.rename(columns={'skewness_premium': 'Average Skewness'}, inplace=True)

        return avg_skewness_data

    def calculate_farthest_otm_skewness_multiple_expiries(df):
        """
        Calculate skewness premium for the deepest available out-of-the-money option pairs across multiple expiries.

        :param df: DataFrame containing options data
        :return: DataFrame with skewness premium for each quote_date
        """
        # Sort the DataFrame by 'quote_date', 'expire_date', and 'strike'
        df = df.sort_values(by=['quote_date', 'expire_date', 'strike'])

        # Group by 'quote_date'
        grouped = df.groupby('quote_date')

        def skewness_for_group(group):
            # Initialize storage for skewness premiums across expiries
            skewness_premiums = []

            # Loop through each unique expiry date within the group
            for expiry in group['expire_date'].unique():
                # Filter options for this expiry date
                expiry_options = group[group['expire_date'] == expiry]

                # Sort options by strike price
                expiry_options_sorted = expiry_options.sort_values(by='strike')

                # Find the farthest OTM call and put
                farthest_otm_call = expiry_options_sorted.iloc[-1]  # Last row (highest strike call)
                farthest_otm_put = expiry_options_sorted.iloc[0]  # First row (lowest strike put)

                # Ensure both call and put have valid data
                if pd.isna(farthest_otm_call['C_Avg']) or pd.isna(farthest_otm_put['P_Avg']):
                    continue  # Skip this expiry if the data is invalid

                call_avg = farthest_otm_call['C_Avg']
                put_avg = farthest_otm_put['P_Avg']

                # Skip calculation if call_avg is zero to avoid division by zero
                if call_avg == 0:
                    continue

                # Calculate skewness premium for this expiry
                skewness_premium = (put_avg / call_avg) - 1
                skewness_premiums.append(skewness_premium)

            # If no valid skewness premiums were calculated, return None
            if len(skewness_premiums) == 0:
                return None
            skewness_premiums.sort()
            lower_percentile = 5
            upper_percentile = 95
            # Remove outliers based on percentile thresholds
            lower_bound = np.percentile(skewness_premiums, lower_percentile)
            upper_bound = np.percentile(skewness_premiums, upper_percentile)

            # Filter out the skewness premiums that are outside the bounds
            filtered_skewness_premiums = [sp for sp in skewness_premiums if lower_bound <= sp <= upper_bound]

            # Calculate the average skewness premium across expiries
            avg_skewness_premium = sum(filtered_skewness_premiums) / len(filtered_skewness_premiums)

            return avg_skewness_premium

        # Apply the skewness calculation for each quote_date
        skewness_data = grouped.apply(skewness_for_group).reset_index()
        skewness_data.columns = ['quote_date', 'skewness_premium']

        return skewness_data

    def calculate_average_skewness_same_strike_same_dte(df):
        """
        Calculate skewness premium using the deepest available out-of-the-money option pairs across multiple expiries for each quote_date.

        :param df: DataFrame containing options data (already filtered for DTE > 28, DTE < 200, and valid premiums).
        :return: DataFrame with skewness premium for each quote_date.
        """

        def get_deepest_available_strikes(group):
            """
            Find the deepest available strike for both calls and puts that are available across all expiries.
            """
            # Sort by strike
            group_sorted = group.sort_values(by='strike')

            # Find the deepest call (highest strike) and deepest put (lowest strike)
            deepest_call_strike = group_sorted['strike'].max()
            deepest_put_strike = group_sorted['strike'].min()

            # Get the rows for the deepest call and put across all expiries
            call_options = group[(group['strike'] == deepest_call_strike)]
            put_options = group[(group['strike'] == deepest_put_strike)]

            if call_options.empty or put_options.empty:
                return None

            # Return the average prices for both call and put, averaging across expiries
            call_avg = call_options['C_Avg'].median()
            put_avg = put_options['P_Avg'].median()
            # call_avg_ = call_options['C_Avg'].mean()
            # put_avg_ = put_options['P_Avg'].mean()

            return call_avg, put_avg

        # Group by 'quote_date'
        grouped = df.groupby('quote_date')

        def calculate_skewness_for_group(group):
            # Get the deepest available strike for this quote_date
            result = get_deepest_available_strikes(group)
            if result is None:
                return None

            call_avg, put_avg = result

            # Ensure both call and put have valid averages
            if call_avg > 0 and put_avg > 0:
                # Calculate skewness premium
                skewness_premium = (put_avg / call_avg) - 1
                return skewness_premium
            return None

        # Apply skewness calculation for each quote_date
        skewness_data = grouped.apply(calculate_skewness_for_group).reset_index()
        skewness_data.columns = ['quote_date', 'skewness_premium']

        return skewness_data

    # skewness_data = calculate_skewness_for_nearest_expiry(filtered_data)
    # skewness_data = calculate_skewness_for_multiple_expiries(filtered_data)
    # skewness_data = calculate_farthest_otm_skewness_multiple_expiries(filtered_data)
    skewness_data = calculate_average_skewness_same_strike_same_dte(filtered_data)
    print(f"Skewness calculation took {time.time() - skewness_calc_start:.2f} seconds")

    # Apply the function to each group
    # skewness_data = grouped.apply(calculate_skewness_for_group).reset_index()
    # skewness_data.columns = ['quote_date', 'skewness_premium']
    # Display the first few rows to check the result
    # print(skewness_data)
    avg_skew_start = time.time()

    # Drop the EXPIRE_DATE column as it's no longer relevant
    skewness_data = skewness_data.drop(columns='expire_date', errors='ignore')
    # print(skewness_data)
    # Calculate the average skewness premium for each QUOTE_DATE
    average_skewness = skewness_data.groupby('quote_date').max().reset_index()
    # Rename the second column to 'Average Skewness'
    average_skewness.rename(columns={"skewness_premium": 'Average Skewness'}, inplace=True)
    # print(average_skewness)
    # print('average skew')
    # print(average_skewness)
    print(f"Averaging skewness took {time.time() - avg_skew_start:.2f} seconds")


    def q_gaussian(x, mu, sigma, q):
        B_q = (3 - q) * sigma ** 2
        A_q = np.sqrt((q - 1) / (np.pi * B_q))
        return A_q * (1 - (1 - q) * B_q * (x - mu) ** 2) ** (1 / (1 - q))

    def tsallis_entropy(prob_dist, q):
        if q != 1:
            entropy = (1 - np.sum(prob_dist ** q)) / (q - 1)
        else:
            entropy = -np.sum(prob_dist * np.log(prob_dist))
        return entropy

    # Parameters
    mu_q = average_skewness['Average Skewness'].mean()
    sigma_q = average_skewness['Average Skewness'].std()
    q = 1.4  # Example q value

    # Calculate the q-Gaussian distribution for the data
    # prob_dist = q_gaussian(average_skewness['Average Skewness'], mu_q, sigma_q, q)
    # print(prob_dist)
    # Normalize the probability distribution
    # prob_dist /= prob_dist.sum()

    # entropy_value = tsallis_entropy(prob_dist, q)
    # print(f"Tsallis Entropy: {entropy_value}")


    def _q_gaussian_log_likelihood(q, data, mu, sigma):
        """
        Calculate the negative log-likelihood of the q-Gaussian distribution.
        """
        B_q = (3 - q) * sigma ** 2
        A_q = np.sqrt((q - 1) / (np.pi * B_q))
        p_x = A_q * (1 - (1 - q) * B_q * (data - mu) ** 2) ** (1 / (1 - q))

        # Ensure that p_x is positive and valid for log calculation
        p_x = np.clip(p_x, 1e-10, None)

        # Return the negative log-likelihood (since we minimize, not maximize)
        return -np.sum(np.log(p_x))


    # def _estimate_q(data, mu, sigma, initial_q=1.2):
    #     """
    #     Estimate the optimal q using MLE.
    #     """
    #     result = minimize(q_gaussian_log_likelihood, initial_q, args=(data, mu, sigma),
    #                       bounds=[(1, 3)])  # q is typically between 1 and 3
    #
    #     if result.success:
    #         return result.x[0]
    #     else:
    #         raise ValueError("Optimization failed to find the optimal q.")

    # Estimate q
    # estimated_q = estimate_q(average_skewness["Average Skewness"], mu_q, sigma_q)
    # print(f"Estimated q: {estimated_q}")

    K = 50  # Window size
    Delta = 1  # Sliding step
    q_x = 1.62  # Given q value
    # q_x = 2  # Given q value
    skewness_data = average_skewness['Average Skewness']
    quote_dates = average_skewness['quote_date']
    # print(skewness_data)
    # print(average_skewness)
    def calculate_time_dependent_entropy_with_skewness(skewness_data, quote_dates, window_width, sliding_step, q,
                                                       num_partitions=10):
        """
        Calculate the time-dependent Tsallis entropy using a sliding window technique.

        Parameters:
        - signal: The time series data (e.g., skewness premiums).
        - window_width: The width of the sliding window (w).
        - sliding_step: The step size for moving the window (Δ).
        - q: The Tsallis entropy parameter.
        - num_partitions: The number of partitions (intervals) for the data within each window.

        Returns:
        - entropy_values: A DataFrame containing the entropy values and corresponding dates.
        """
        entropy_values = []
        aligned_skewness = []
        dates = []

        num_windows = (len(skewness_data) - window_width) // sliding_step + 1

        for n in range(num_windows):
            start_index = n * sliding_step
            end_index = start_index + window_width
            window_data = skewness_data[start_index:end_index]

            # Partition the data into `num_partitions` intervals
            min_val, max_val = window_data.min(), window_data.max()
            intervals = np.linspace(min_val, max_val, num_partitions + 1)
            digitized = np.digitize(window_data, intervals) - 1  # Get interval indices

            # Calculate probabilities for each interval
            prob_dist = np.array([(digitized == i).sum() for i in range(num_partitions)]) / len(window_data)

            prob_dist = prob_dist[prob_dist > 0]

            # Calculate entropy
            entropy = tsallis_entropy(prob_dist, q)
            entropy_values.append(entropy)

            # Store the date associated with the current window's end point
            dates.append(quote_dates.iloc[end_index - 1])

            # Align the skewness data with the entropy calculation
            aligned_skewness.append(window_data.iloc[-1])

        # Create DataFrame for easier analysis
        entropy_df = pd.DataFrame({
            'Date': dates,
            'Entropy': entropy_values,
            'Skewness': aligned_skewness
        })

        return entropy_df

    # print('entropy_df')
    # print(entropy_df)
    # return calculate_time_dependent_entropy_with_skewness(skewness_data, quote_dates, K, Delta, q_x)
    def entropy():
        entropy_values = []
        aligned_skewness = []
        dates = []

        # Adjust loop to ensure no forward-looking data is used
        for end in range(K, len(average_skewness) + 1, Delta):
            # Define the window to use only past and present data
            window_data = skewness_data[end - K:end]  # Window includes K values up to the current point

            # Calculate mean and standard deviation for the window
            mu_q = np.mean(window_data)
            sigma_q = np.std(window_data)

            # Calculate the q-Gaussian distribution for the window (using skewness data directly)
            prob_dist = q_gaussian(window_data, mu_q, sigma_q, q_x)
            prob_dist /= np.sum(prob_dist)  # Normalize the probability distribution

            # Calculate Tsallis Entropy for the window
            entropy_value = tsallis_entropy(prob_dist, q_x)
            entropy_values.append(entropy_value)

            # Store the date corresponding to the end of the window
            dates.append(quote_dates.iloc[end - 1])
            aligned_skewness.append(window_data.iloc[-1])

        # Convert to DataFrame for easier plotting and analysis
        entropy_df = pd.DataFrame({
            'Date': dates,
            'Entropy': entropy_values,
            'Skewness': aligned_skewness
        })
        # print('entropy df')
        # print(entropy_df)
        return entropy_df
    # return entropy()
    entropy_calc_start = time.time()

    def calculate_approximate_entropy_with_skewness(skewness_data, quote_dates, window_width=K, sliding_step=Delta, m=2,
                                                    r=None):
        """
        Calculate Approximate Entropy (ApEn) for a given time series with aligned dates and skewness using a sliding window.

        Parameters:
        - skewness_data: The time series data (e.g., skewness premiums).
        - quote_dates: The list of corresponding quote dates.
        - window_width: The width of the sliding window.
        - sliding_step: The step size for moving the window.
        - m: The dimension of the vectors u(m)(i) used in the ApEn calculation. Default is 2.
        - r: The tolerance level. If not provided, it will be calculated as r = 0.15 * std_dev of the data.

        Returns:
        - entropy_df: A DataFrame containing the Approximate Entropy, corresponding dates, and skewness.
        """
        N = len(skewness_data)
        if N < m + 1:
            raise ValueError("Time series is too short to calculate approximate entropy.")

        # Calculate standard deviation of the time series
        std_dev = np.std(skewness_data)

        # Set tolerance r if not provided
        if r is None:
            r = 0.15 * std_dev

        # Store results
        entropy_values = []
        aligned_skewness = []
        aligned_dates = []

        # Step 1: Create m-dimensional vectors
        def create_m_dimensional_vectors(data, m):
            vectors = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
            return vectors

        # Step 2: Calculate distance between vectors
        def max_distance(v1, v2):
            return np.max(np.abs(v1 - v2))

        # Step 3: Calculate C(m)(u(m)(i)|X, r)
        def calculate_C_m(vectors, r):
            N_m = len(vectors)
            C_m = np.zeros(N_m)
            for i in range(N_m):
                for j in range(N_m):
                    if max_distance(vectors[i], vectors[j]) <= r:
                        C_m[i] += 1
                C_m[i] /= N_m
            return C_m

        # Step 4: Calculate Φ(m)(r)
        def calculate_Phi(C_m):
            return np.sum(np.log(C_m)) / len(C_m)

        # Sliding window over the skewness data
        num_windows = (N - window_width) // sliding_step + 1

        for n in range(num_windows):
            start_index = n * sliding_step
            end_index = start_index + window_width
            window_data = skewness_data[start_index:end_index]

            # Create m and (m+1)-dimensional vectors for the current window
            vectors_m = create_m_dimensional_vectors(window_data, m)
            vectors_m1 = create_m_dimensional_vectors(window_data, m + 1)

            # Calculate C(m) and C(m+1)
            C_m = calculate_C_m(vectors_m, r)
            C_m1 = calculate_C_m(vectors_m1, r)

            # Calculate Φ(m) and Φ(m+1)
            Phi_m = calculate_Phi(C_m)
            Phi_m1 = calculate_Phi(C_m1)

            # Calculate Approximate Entropy (ApEn)
            ApEn = Phi_m - Phi_m1
            entropy_values.append(ApEn)

            # Align dates with the entropy calculation (date at the end of the window)
            aligned_dates.append(quote_dates[end_index - 1])

            # Align skewness data with the entropy calculation (last skewness value in the window)
            aligned_skewness.append(window_data.iloc[-1])

        # Return the results in a DataFrame
        entropy_df = pd.DataFrame({
            'Date': aligned_dates,
            'Entropy': entropy_values,
            'Skewness': aligned_skewness
        })
        # print(entropy_df.columns)
        return entropy_df

    print(f"Entropy calculation took {time.time() - entropy_calc_start:.2f} seconds")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    return calculate_approximate_entropy_with_skewness(skewness_data, quote_dates, window_width=50, m=2)
