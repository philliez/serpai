!pip install lightweight-charts


Run The code above ^ once, after opening this project.

Next, run the code cell below this.

import pandas as pd
from datetime import datetime
import asyncio
import json
import time
import re
from google.colab import data_table
from IPython.display import display, HTML
from lightweight_charts import JupyterChart
import gdown  # Added import

# Stop Loss and Take Profit percentages
STOP_LOSS_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]
TAKE_PROFIT_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]

def get_histo_color(current_value, previous_value, macd):
    """Determines the color of the MACD histogram based on current and previous values."""
    macd_positive = macd > 0
    try:
        current_value = float(current_value)
        previous_value = float(previous_value) if previous_value is not None else float('-inf')
    except (ValueError, TypeError):
        return ""

    if current_value > previous_value:
        return "Dark Green" if macd_positive else "Red"
    elif current_value < previous_value:
        return "Light Green" if macd_positive else "Pink"
    else:
        return ""

def calculate_score(color, macd_pos_neg):
    """Calculates a score based on the histogram color and MACD direction."""
    if color == "Pink" and macd_pos_neg == "Negative": return 10
    elif color == "Light Green" and macd_pos_neg == "Negative": return 20
    elif color == "Dark Green" and macd_pos_neg == "Negative": return 30
    elif color == "Dark Green" and macd_pos_neg == "Positive": return 40
    elif color == "Light Green" and macd_pos_neg == "Positive": return 30
    elif color == "Pink" and macd_pos_neg == "Positive": return 0
    elif color == "Red" and macd_pos_neg == "Positive": return -10
    elif color == "Red" and macd_pos_neg == "Negative": return -20
    else: return 0

def find_closest_previous_time_binary(target_time, data):
    """Finds the closest previous time in the data using binary search."""
    if data.empty:
        return None
    target_time = pd.to_datetime(target_time, utc=True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)
    prior_data = data[data['Timestamp'] <= target_time]
    if prior_data.empty:
        return None
    closest_time = prior_data.iloc[prior_data['Timestamp'].sub(target_time).abs().argsort()[:1]]
    return closest_time

def calculate_max_profit(price_series, entry_price, stop_loss_levels, take_profit_levels):
    """Calculates the maximum profit based on stop loss and take profit levels."""
    max_profits = {sl: 0 for sl in stop_loss_levels}
    stop_loss_hit = {sl: False for sl in stop_loss_levels}
    take_profit_hit = {tp: False for tp in take_profit_levels}
    exit_price = None
    exit_reason = None
    highest_gain_before_exit = 0


    for price in price_series:
        percent_change = ((price - entry_price) / entry_price)
        highest_gain_before_exit = max(highest_gain_before_exit, percent_change)

        for sl in stop_loss_levels:
            if not stop_loss_hit[sl] and percent_change <= -sl:
                stop_loss_hit[sl] = True
                exit_price = price
                exit_reason = f'Stop Loss ({sl*100:.1f}%)'
                break  # Exit the inner loop (stop loss levels) after a stop loss is hit

        for tp in take_profit_levels:
            if not take_profit_hit[tp] and percent_change >= tp:
                take_profit_hit[tp] = True
                exit_price = price
                exit_reason = f'Take Profit ({tp*100:.1f}%)'
                break  # Exit the inner loop (take profit levels)


        if exit_price is not None:
            break # Exit outer loop when an exit condition (stop loss or take profit) happens

    results = {
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Exit Reason': exit_reason,
        'Highest Gain Before Exit': highest_gain_before_exit * 100, # Percentage
    }
    for sl in stop_loss_levels:
        results[f'Stop Loss Hit ({sl*100:.1f}%)'] = str(stop_loss_hit[sl]) # Convert to string

    for tp in take_profit_levels:
        results[f'Take Profit Hit ({tp*100:.1f}%)'] = str(take_profit_hit[tp]) # Convert to string

    return results

def create_lightweight_chart(df, title="Candlestick Chart with Score"):
    """Creates a Lightweight Chart with candlestick and score series."""

    chart = JupyterChart(width=1000, height=600)

    # Convert 'Timestamp' to datetime if it isn't already
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert datetime to timestamp for the chart
    df['time'] = df['Timestamp'].astype('datetime64[ms]')

    # Add candlestick series
    candlestick_data = df[['time', 'Open', 'High', 'Low', 'Close']].rename(
        columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
    )
    chart.set(candlestick_data)

    # Add score series
    # Ensure 'Score' exists in the DataFrame passed to this function
    if 'Score' in df.columns:
        score_data = df[['time', 'Score']].rename(columns={'Score': 'value'})
        line = chart.create_line(name='Score', color='blue')
        line.set(score_data)
    else:
        print("Warning: 'Score' column not found in DataFrame. Score series will not be plotted.")

    return chart

def process_data(file_path):
    """Processes data from CSV, calculates scores, and stop loss results."""
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce', utc=True)
        df.dropna(subset=['Timestamp'], inplace=True)
        df = df.rename(columns={'Histogram': 'Histo', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

        # Added fix for timezone-aware to timezone-naive conversion
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

        total_rows = len(df)
        print(f"Processing {total_rows} rows from {file_path}")

        df['Histo Color'] = ""
        df['Histo +/-'] = ""
        df['Histo Value'] = 0.0
        # Ensure 'Score' is a numeric column, initialized with 0
        df['Score'] = 0  # Or you could use np.nan if missing values are okay initially

        for index, row in df.iterrows():
            if (index + 1) % 50 == 0:
                print(f"Processing Score Data: {index + 1}/{total_rows} rows")

            try:
                previous_histo = df[df['Timestamp'] < row['Timestamp']]['Histo'].iloc[-1]
            except IndexError:
                previous_histo = None
            macd_value = float(row['MACD'])
            histo_color = get_histo_color(row['Histo'], previous_histo, macd_value)
            macd_pos_neg = "Positive" if macd_value >= 0 else "Negative"
            score = calculate_score(histo_color, macd_pos_neg)

            df.loc[index, ['Histo Color', 'Histo +/-']] = histo_color, macd_pos_neg
            df.loc[index, 'Histo Value'] = float(row['Histo']) if row['Histo'] is not None else 0.0
            df.loc[index, 'Score'] = int(score)  # Make SURE 'score' is a single number (int or float)

        print("Calculating Stop Loss/Take Profit Results...")
        # Calculate Stop Loss/Take Profit Results
        results_list = []
        for index, row in df.iterrows():
            if (index + 1) % 50 == 0:
                print(f"Calculating Stop Loss for {index + 1}/{total_rows} rows")
            entry_price = row['Open']
            price_series = df[df['Timestamp'] >= row['Timestamp']]['Close'].tolist()  # Include current candle and future
            stop_loss_results = calculate_max_profit(price_series, entry_price, STOP_LOSS_PERCENTAGES, TAKE_PROFIT_PERCENTAGES)
            row_results = row.to_dict()
            row_results.update(stop_loss_results)
            results_list.append(row_results)

        results_df = pd.DataFrame(results_list)

        print("Saving processed data to CSV...")
        # Save to CSV (optional, but good practice for debugging)
        results_df.to_csv("processed_data.csv", index=False)
        print("Processed data saved to processed_data.csv")

        print("Generating chart...")
        # Generate chart
        # Use the original dataframe for the chart generation
        # Before sending to create_lightweight_chart print head
        print("DataFrame Columns before chart:", df.columns)  # *** ADDED DEBUGGING PRINT ***
        print(df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Score']].head())  # *** ADDED DEBUGGING PRINT ***

        chart = create_lightweight_chart(df)
        chart.show()

        print("Displaying datatable...")
        # Display Datatable
        display(data_table.DataTable(results_df, num_rows_per_page=20))  # Display the DataFrame as an interactive table
        print("Finished processing.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def main():
    # Download the file from Google Drive
    file_id = "1yPsQtD4YhhZF2XbWg5pPhEni5cdVHyF3"  # Replace with your Google Drive file ID
    output_path = "XLMUSDT.csv"  # Local filename to save the downloaded file

    try:
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
        # Process the downloaded data
        process_data(output_path)

    except Exception as e:
        print(f"Error downloading or processing file: {e}")


if __name__ == '__main__':
    # Install gdown if it's not already installed
    try:
        import gdown
    except ImportError:
        print("gdown not found. Installing...")
        !pip install gdown
        import gdown

    # Run the main function
    main()

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from datetime import datetime
import asyncio
import json
import time
import re
from google.colab import data_table
from IPython.display import display, HTML
from lightweight_charts import JupyterChart
import gdown  # Added import

# Stop Loss and Take Profit percentages
STOP_LOSS_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]
TAKE_PROFIT_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]

def get_histo_color(current_value, previous_value, macd):
    """Determines the color of the MACD histogram based on current and previous values."""
    macd_positive = macd > 0
    try:
        current_value = float(current_value)
        previous_value = float(previous_value) if previous_value is not None else float('-inf')
    except (ValueError, TypeError):
        return ""

    if current_value > previous_value:
        return "Dark Green" if macd_positive else "Red"
    elif current_value < previous_value:
        return "Light Green" if macd_positive else "Pink"
    else:
        return ""

def calculate_score(color, macd_pos_neg):
    """Calculates a score based on the histogram color and MACD direction."""
    if color == "Pink" and macd_pos_neg == "Negative": return 10
    elif color == "Light Green" and macd_pos_neg == "Negative": return 20
    elif color == "Dark Green" and macd_pos_neg == "Negative": return 30
    elif color == "Dark Green" and macd_pos_neg == "Positive": return 40
    elif color == "Light Green" and macd_pos_neg == "Positive": return 30
    elif color == "Pink" and macd_pos_neg == "Positive": return 0
    elif color == "Red" and macd_pos_neg == "Positive": return -10
    elif color == "Red" and macd_pos_neg == "Negative": return -20
    else: return 0

def find_closest_previous_time_binary(target_time, data):
    """Finds the closest previous time in the data using binary search."""
    if data.empty:
        return None
    target_time = pd.to_datetime(target_time, utc=True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)
    prior_data = data[data['Timestamp'] <= target_time]
    if prior_data.empty:
        return None
    closest_time = prior_data.iloc[prior_data['Timestamp'].sub(target_time).abs().argsort()[:1]]
    return closest_time

def calculate_max_profit(price_series, entry_price, stop_loss_levels, take_profit_levels):
    """Calculates the maximum profit based on stop loss and take profit levels."""
    max_profits = {sl: 0 for sl in stop_loss_levels}
    stop_loss_hit = {sl: False for sl in stop_loss_levels}
    take_profit_hit = {tp: False for tp in take_profit_levels}
    exit_price = None
    exit_reason = None
    highest_gain_before_exit = 0


    for price in price_series:
        percent_change = ((price - entry_price) / entry_price)
        highest_gain_before_exit = max(highest_gain_before_exit, percent_change)

        for sl in stop_loss_levels:
            if not stop_loss_hit[sl] and percent_change <= -sl:
                stop_loss_hit[sl] = True
                exit_price = price
                exit_reason = f'Stop Loss ({sl*100:.1f}%)'
                break  # Exit the inner loop (stop loss levels) after a stop loss is hit

        for tp in take_profit_levels:
            if not take_profit_hit[tp] and percent_change >= tp:
                take_profit_hit[tp] = True
                exit_price = price
                exit_reason = f'Take Profit ({tp*100:.1f}%)'
                break  # Exit the inner loop (take profit levels)


        if exit_price is not None:
            break # Exit outer loop when an exit condition (stop loss or take profit) happens

    results = {
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Exit Reason': exit_reason,
        'Highest Gain Before Exit': highest_gain_before_exit * 100, # Percentage
    }
    for sl in stop_loss_levels:
        results[f'Stop Loss Hit ({sl*100:.1f}%)'] = str(stop_loss_hit[sl]) # Convert to string

    for tp in take_profit_levels:
        results[f'Take Profit Hit ({tp*100:.1f}%)'] = str(take_profit_hit[tp]) # Convert to string

    return results

def create_lightweight_chart(df, title="Candlestick Chart with Score"):
    """Creates a Lightweight Chart with candlestick and score series."""

    chart = JupyterChart(width=1000, height=600)

    # Convert 'Timestamp' to datetime if it isn't already
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert datetime to timestamp for the chart
    df['time'] = df['Timestamp'].astype('datetime64[ms]')

    # Add candlestick series
    candlestick_data = df[['time', 'Open', 'High', 'Low', 'Close']].rename(
        columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
    )
    chart.set(candlestick_data)

    # Add score series
    # Ensure 'Score' exists in the DataFrame passed to this function
    if 'Score' in df.columns:
        score_data = df[['time', 'Score']].rename(columns={'Score': 'value'})
        line = chart.create_line(name='Score', color='blue')
        line.set(score_data)
    else:
        print("Warning: 'Score' column not found in DataFrame. Score series will not be plotted.")

    return chart

def process_data(file_path):
    """Processes data from CSV, calculates scores, and stop loss results."""
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce', utc=True)
        df.dropna(subset=['Timestamp'], inplace=True)
        df = df.rename(columns={'Histogram': 'Histo', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

        # Added fix for timezone-aware to timezone-naive conversion
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

        total_rows = len(df)
        print(f"Processing {total_rows} rows from {file_path}")

        df['Histo Color'] = ""
        df['Histo +/-'] = ""
        df['Histo Value'] = 0.0
        # Ensure 'Score' is a numeric column, initialized with 0
        df['Score'] = 0  # Or you could use np.nan if missing values are okay initially

        for index, row in df.iterrows():
            if (index + 1) % 50 == 0:
                print(f"Processing Score Data: {index + 1}/{total_rows} rows")

            try:
                previous_histo = df[df['Timestamp'] < row['Timestamp']]['Histo'].iloc[-1]
            except IndexError:
                previous_histo = None
            macd_value = float(row['MACD'])
            histo_color = get_histo_color(row['Histo'], previous_histo, macd_value)
            macd_pos_neg = "Positive" if macd_value >= 0 else "Negative"
            score = calculate_score(histo_color, macd_pos_neg)

            df.loc[index, ['Histo Color', 'Histo +/-']] = histo_color, macd_pos_neg
            df.loc[index, 'Histo Value'] = float(row['Histo']) if row['Histo'] is not None else 0.0
            df.loc[index, 'Score'] = int(score)  # Make SURE 'score' is a single number (int or float)

        print("Calculating Stop Loss/Take Profit Results...")
        # Calculate Stop Loss/Take Profit Results
        results_list = []
        for index, row in df.iterrows():
            if (index + 1) % 50 == 0:
                print(f"Calculating Stop Loss for {index + 1}/{total_rows} rows")
            entry_price = row['Open']
            price_series = df[df['Timestamp'] >= row['Timestamp']]['Close'].tolist()  # Include current candle and future
            stop_loss_results = calculate_max_profit(price_series, entry_price, STOP_LOSS_PERCENTAGES, TAKE_PROFIT_PERCENTAGES)
            row_results = row.to_dict()
            row_results.update(stop_loss_results)
            results_list.append(row_results)

        results_df = pd.DataFrame(results_list)

        print("Saving processed data to CSV...")
        # Save to CSV (optional, but good practice for debugging)
        results_df.to_csv("processed_data.csv", index=False)
        print("Processed data saved to processed_data.csv")

        print("Generating chart...")
        # Generate chart
        # Use the original dataframe for the chart generation
        # Before sending to create_lightweight_chart print head
        print("DataFrame Columns before chart:", df.columns)  # *** ADDED DEBUGGING PRINT ***
        print(df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Score']].head())  # *** ADDED DEBUGGING PRINT ***

        chart = create_lightweight_chart(df)
        chart.show()

        print("Displaying datatable...")
        # Display Datatable
        display(data_table.DataTable(results_df, num_rows_per_page=20))  # Display the DataFrame as an interactive table
        print("Finished processing.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def main():
    # Download the file from Google Drive
    file_id = "1yPsQtD4YhhZF2XbWg5pPhEni5cdVHyF3"  # Replace with your Google Drive file ID
    output_path = "XLMUSDT.csv"  # Local filename to save the downloaded file

    try:
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
        # Process the downloaded data
        process_data(output_path)

    except Exception as e:
        print(f"Error downloading or processing file: {e}")


if __name__ == '__main__':
    # Install gdown if it's not already installed
    try:
        import gdown
    except ImportError:
        print("gdown not found. Installing...")
        !pip install gdown
        import gdown

    # Run the main function
    main()