import pandas as pd
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# Stop Loss and Take Profit percentages - moved from Scoring.py
STOP_LOSS_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]
TAKE_PROFIT_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]

# Helper functions - moved from Scoring.py
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

def calculate_max_profit(price_series, entry_price, stop_loss_levels, take_profit_levels):
    """Calculates the maximum profit based on stop loss and take profit levels."""
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
                break

        for tp in take_profit_levels:
            if not take_profit_hit[tp] and percent_change >= tp:
                take_profit_hit[tp] = True
                exit_price = price
                exit_reason = f'Take Profit ({tp*100:.1f}%)'
                break

        if exit_price is not None:
            break

    results = {
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Exit Reason': exit_reason,
        'Highest Gain Before Exit': highest_gain_before_exit * 100,
    }
    for sl in stop_loss_levels:
        results[f'Stop Loss Hit ({sl*100:.1f}%)'] = str(stop_loss_hit[sl])

    for tp in take_profit_levels:
        results[f'Take Profit Hit ({tp*100:.1f}%)'] = str(take_profit_hit[tp])

    return results

def process_data_for_api(file_object):
    """Processes data from CSV, calculates scores, and stop loss results for API."""
    try:
        df = pd.read_csv(file_object)
        df['Timestamp'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce', utc=True)
        df.dropna(subset=['Timestamp'], inplace=True)
        df = df.rename(columns={'Histogram': 'Histo', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

        # Added fix for timezone-aware to timezone-naive conversion
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

        df['Histo Color'] = ""
        df['Histo +/-'] = ""
        df['Histo Value'] = 0.0
        df['Score'] = 0

        for index, row in df.iterrows():
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
            df.loc[index, 'Score'] = int(score)

        results_list = []
        for index, row in df.iterrows():
            entry_price = row['Open']
            price_series = df[df['Timestamp'] >= row['Timestamp']]['Close'].tolist()
            stop_loss_results = calculate_max_profit(price_series, entry_price, STOP_LOSS_PERCENTAGES, TAKE_PROFIT_PERCENTAGES)
            row_results = row.to_dict()
            row_results.update(stop_loss_results)
            results_list.append(row_results)

        results_df = pd.DataFrame(results_list)

        # Prepare candlestick data for lightweight-charts
        candlestick_data = df[['Timestamp', 'Open', 'High', 'Low', 'Close']].rename(
            columns={'Timestamp': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
        )
        # Convert datetime to timestamp for the chart (Lightweight Charts expects Unix timestamps in seconds or milliseconds)
        candlestick_data['time'] = candlestick_data['time'].astype('int64') // 10**9 # Unix timestamp in seconds

        # Prepare score data for lightweight-charts
        score_data = df[['Timestamp', 'Score']].rename(columns={'Timestamp': 'time', 'Score': 'value'})
        score_data['time'] = score_data['time'].astype('int64') // 10**9 # Unix timestamp in seconds

        return {
            "processed_data": results_df.to_dict(orient='records'),
            "chart_data": {
                "candlestick": candlestick_data.to_dict(orient='records'),
                "score_line": score_data.to_dict(orient='records')
            }
        }

    except Exception as e:
        raise Exception(f"Error processing data: {e}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            # Read the file directly into pandas
            data = process_data_for_api(io.StringIO(file.read().decode('utf-8')))
            return jsonify({"message": "File processed successfully", "data": data}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)