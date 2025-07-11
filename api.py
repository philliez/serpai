import pandas as pd
import numpy as np
import asyncio
import json
import base64
import io
import os
import logging
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify
from playwright.async_api import async_playwright

# --- Basic Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Preset Management for Render Deployment ---
# Use Render's persistent disk mount path via an environment variable.
# Fallback to the current directory for local development.
DATA_DIR = os.environ.get('DATA_DIR', '.')
PRESETS_FILE = os.path.join(DATA_DIR, 'presets.json')

# Ensure the data directory exists when the app starts.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logging.info(f"Created data directory at: {DATA_DIR}")

def load_presets():
    """Loads presets from the JSON file on the persistent disk."""
    if not os.path.exists(PRESETS_FILE):
        return {"last_run": {}, "history": []}
    try:
        with open(PRESETS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading presets: {e}")
        return {"last_run": {}, "history": []}

def save_presets(params):
    """Saves the latest run parameters and updates history."""
    presets = load_presets()
    # Use a deque for efficient append/pop operations on a fixed-size list
    history_deque = deque(presets.get("history", []), maxlen=5)

    if presets.get("last_run"):
        # Avoid adding duplicate consecutive runs to history
        if not any(h['run_timestamp'] == presets['last_run']['run_timestamp'] for h in history_deque):
            history_deque.appendleft(presets["last_run"])

    presets["last_run"] = params
    presets["history"] = list(history_deque)

    try:
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=4)
            logging.info(f"Presets saved successfully to {PRESETS_FILE}")
    except IOError as e:
        logging.error(f"Error saving presets: {e}")


# --- Business Logic ---

def get_histo_color(current_value, previous_value, macd):
    """Determines the color of the MACD histogram."""
    try:
        macd_positive = float(macd) > 0
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

def calculate_max_profit(price_series, entry_price, stop_loss_levels, take_profit_levels):
    """Calculates trade outcomes based on stop loss and take profit levels."""
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
                if exit_price is None: # Record the first exit event
                    exit_price = price
                    exit_reason = f'Stop Loss ({sl*100:.1f}%)'

        for tp in take_profit_levels:
            if not take_profit_hit[tp] and percent_change >= tp:
                take_profit_hit[tp] = True
                if exit_price is None: # Record the first exit event
                    exit_price = price
                    exit_reason = f'Take Profit ({tp*100:.1f}%)'

    results = {
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Exit Reason': exit_reason,
        'Highest Gain Before Exit': f"{highest_gain_before_exit * 100:.2f}%",
    }
    for sl in stop_loss_levels:
        results[f'Stop Loss Hit ({sl*100:.1f}%)'] = stop_loss_hit[sl]
    for tp in take_profit_levels:
        results[f'Take Profit Hit ({tp*100:.1f}%)'] = take_profit_hit[tp]

    return results

# --- Server-Side Chart Generation ---

async def generate_chart_screenshot(page, chart_data, entry_marker_time):
    """Renders a chart using Playwright and returns a base64 screenshot."""
    chart_data_json = json.dumps(chart_data)
    entry_time_json = json.dumps(entry_marker_time)

    html_template = f"""
    <!DOCTYPE html><html><head>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    </head><body style="margin:0; background-color:#131722;">
    <div id="chart" style="width: 800px; height: 500px;"></div>
    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            width: 800, height: 500, layout: {{ backgroundColor: '#131722', textColor: '#d1d4dc' }},
            grid: {{ vertLines: {{ color: '#334158' }}, horzLines: {{ color: '#334158' }} }}
        }});
        const candleSeries = chart.addCandlestickSeries({{ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' }});
        const data = {chart_data_json};
        candleSeries.setData(data);
        const entryTime = {entry_time_json};
        candleSeries.setMarkers([{{ time: entryTime, position: 'belowBar', color: '#2196F3', shape: 'arrowUp', text: 'Entry' }}]);
        chart.timeScale().fitContent();
    </script></body></html>
    """

    try:
        await page.set_content(html_template)
        await page.wait_for_selector('#chart', state='attached')
        chart_element = await page.query_selector('#chart')
        screenshot_bytes = await chart_element.screenshot()
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Playwright screenshot error: {e}")
        return None

# --- Main Orchestrator ---

async def process_and_generate_screenshots(df, stop_loss_pct, take_profit_pct):
    """Processes data and orchestrates screenshot generation for each row."""
    processed_rows = []
    screenshots = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        for index, row in df.iterrows():
            # 1. Analyze trade outcomes for the current row
            future_price_series = df.loc[index:, 'Close'].tolist()
            analysis_results = calculate_max_profit(future_price_series, row['Open'], stop_loss_pct, take_profit_pct)
            row_dict = row.to_dict()
            row_dict.update(analysis_results)
            processed_rows.append(row_dict)

            # 2. Prepare data for the chart (e.g., last 60 candles)
            chart_start_index = max(0, index - 59)
            chart_slice = df.iloc[chart_start_index:index + 1]
            chart_data = chart_slice.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
            chart_data['time'] = (chart_data['Timestamp'].astype(np.int64) // 10**9) # Unix timestamp
            chart_data_records = chart_data[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
            entry_timestamp = row['Timestamp'].timestamp()

            # 3. Generate screenshot
            screenshot_b64 = await generate_chart_screenshot(page, chart_data_records, entry_timestamp)
            screenshots.append({"timestamp": row['Timestamp'].isoformat(), "image": screenshot_b64})
            logging.info(f"Generated screenshot for timestamp: {row['Timestamp']}")

        await browser.close()
    return processed_rows, screenshots

# --- API Endpoints ---

@app.route('/presets', methods=['GET'])
def get_presets_endpoint():
    """Endpoint to retrieve saved analysis parameters."""
    return jsonify(load_presets())

@app.route('/process', methods=['POST'])
def process_data_endpoint():
    """Main endpoint to upload a CSV and trigger the analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Parse parameters from the form
        stop_loss_str = request.form.get('stop_loss_percentages', "0.01,0.02,0.05")
        take_profit_str = request.form.get('take_profit_percentages', "0.01,0.02,0.05,0.1")
        stop_loss_pct = [float(x.strip()) for x in stop_loss_str.split(',')]
        take_profit_pct = [float(x.strip()) for x in take_profit_str.split(',')]

        # Read and preprocess the CSV file
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        df['Timestamp'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        df.dropna(subset=['Timestamp'], inplace=True)
        df = df.rename(columns={'Histogram': 'Histo', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # Add MACD color info
        for i, row in df.iterrows():
            prev_histo = df.loc[i - 1, 'Histo'] if i > 0 else None
            df.loc[i, 'Histo Color'] = get_histo_color(row['Histo'], prev_histo, row.get('MACD'))

        # Run the main async processing function
        processed_data, screenshots = asyncio.run(process_and_generate_screenshots(df, stop_loss_pct, take_profit_pct))

        # Save the parameters of this successful run
        params_to_save = {
            "stop_loss_percentages": stop_loss_pct,
            "take_profit_percentages": take_profit_pct,
            "run_timestamp": datetime.now().isoformat()
        }
        save_presets(params_to_save)

        # Prepare final dataframe for JSON serialization
        df_json_ready = pd.DataFrame(processed_data)
        for col in df_json_ready.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
            df_json_ready[col] = df_json_ready[col].dt.isoformat()
        for col in df_json_ready.select_dtypes(include=['bool']).columns:
            df_json_ready[col] = df_json_ready[col].astype(str)
        df_json_ready = df_json_ready.replace({np.nan: None})

        return jsonify({
            "message": "Processing complete.",
            "analysis_results": df_json_ready.to_dict(orient='records'),
            "screenshots": screenshots
        })

    except Exception as e:
        logging.error(f"A top-level error occurred during processing: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Check logs for details."}), 500

if __name__ == '__main__':
    # This block is for local development only.
    # On Render, Gunicorn will be used as defined in render.yaml.
    logging.info("Starting Flask development server...")
    logging.info("Reminder: For the first local run, you may need to run 'playwright install' in your terminal.")
    app.run(host='0.0.0.0', port=5555, debug=True)