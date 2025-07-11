import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# --- Helper functions from your second script ---

def exit_price(row, df, tp, sl):
    """
    Calculates the exit price and result of a trade based on TP/SL levels.
    """
    entry_price = row['open']
    entry_time = row['time']
    tp_price = entry_price * tp
    sl_price = entry_price * sl

    # Get the index of the current entry row to slice the DataFrame for future prices
    idx = df.index.get_loc(row.name)
    future_df = df.iloc[idx+1:]

    for i, next_row in future_df.iterrows():
        if next_row['high'] >= tp_price:
            return pd.Series({
                'Open_price': entry_price,
                'TP_price': tp_price,
                'Exit_price': tp_price,
                'W/L': "WIN",
                'open_datetime': entry_time,
                'end_datetime': next_row['time']
            })
        if next_row['low'] <= sl_price:
            return pd.Series({
                'Open_price': entry_price,
                'SL_price': sl_price,
                'Exit_price': sl_price,
                'W/L': "LOSS",
                'open_datetime': entry_time,
                'end_datetime': next_row['time']
            })

    # If neither TP nor SL is hit by the end of the data
    return pd.Series({
        'Open_price': entry_price,
        'W/L': "OPEN",
        'open_datetime': entry_time,
        'end_datetime': None,
        'Exit_price': None
    })

def process_trade_analysis(file_object, tp, sl):
    """
    Processes the uploaded CSV to perform trade analysis.
    """
    try:
        # Load data and ensure correct data types
        df = pd.read_csv(file_object)
        df['time'] = pd.to_datetime(df['time']) # Assuming 'time' column exists
        df = df.sort_values('time').reset_index(drop=True)

        # Apply the analysis function
        analysis_results = df.apply(exit_price, axis=1, df=df, tp=tp, sl=sl)

        # Combine original data with analysis
        results_df = pd.concat([df, analysis_results], axis=1)

        # Calculate summary
        win_count = (results_df['W/L'] == "WIN").sum()
        loss_count = (results_df['W/L'] == "LOSS").sum()

        # Convert datetime objects to strings for JSON compatibility
        results_df['open_datetime'] = results_df['open_datetime'].astype(str)
        results_df['end_datetime'] = results_df['end_datetime'].astype(str)
        results_df.replace({pd.NaT: None}, inplace=True)


        return {
            "summary": {
                "win_count": int(win_count),
                "loss_count": int(loss_count),
                "win_rate": (win_count / (win_count + loss_count)) if (win_count + loss_count) > 0 else 0,
                "take_profit_multiplier": tp,
                "stop_loss_multiplier": sl
            },
            "trades": results_df.to_dict(orient='records')
        }

    except Exception as e:
        # Return a specific error message
        raise Exception(f"Error processing data: {e}")


# --- API Endpoint ---

@app.route('/analyze_trades', methods=['POST'])
def analyze_trades_endpoint():
    # Check for file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Get TP/SL from form, with default values if not provided
        tp = float(request.form.get('tp', 1.006))
        sl = float(request.form.get('sl', 0.9994))

        # Process the file
        string_io = io.StringIO(file.read().decode('utf-8'))
        analysis_data = process_trade_analysis(string_io, tp=tp, sl=sl)

        return jsonify({
            "message": "Analysis successful",
            "data": analysis_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on port 5555 as in the original request
    app.run(host='localhost', port=5555, debug=True)