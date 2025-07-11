import pandas as pd
import numpy as np
import asyncio
import json
import io
import os
import logging
import uuid
from flask import Flask, request, jsonify, send_file
from playwright.async_api import async_playwright

# --- Basic Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- In-Memory Cache for Analysis Data ---
# This dictionary will hold the DataFrames from recent analyses,
# keyed by a unique ID. This avoids saving files to disk.
ANALYSIS_CACHE = {}
MAX_CACHE_SIZE = 10 # Limit cache size to avoid memory issues

# --- Server-Side Chart Generation ---
async def create_chart_screenshot(chart_data, trade_details):
    """
    Generates a chart screenshot in-memory using Playwright.
    Returns the raw image bytes.
    """
    browser = await async_playwright().start()
    chromium = browser.chromium
    browser_instance = await chromium.launch()
    page = await browser_instance.newPage()
    await page.set_viewport_size({"width": 1000, "height": 600})

    # Convert data to JSON for the HTML template
    chart_json = chart_data.to_json(orient='records')
    entry_time_unix = int(pd.to_datetime(trade_details['open_datetime']).timestamp())
    entry_price = trade_details['Open_price']

    html_content = f"""
    <!DOCTYPE html><html><head>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    </head><body style="margin:0; background-color:#131722;">
    <div id="chart"></div>
    <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            width: 1000, height: 600, layout: {{ backgroundColor: '#131722', textColor: '#d1d4dc' }},
            grid: {{ vertLines: {{ color: '#334158' }}, horzLines: {{ color: '#334158' }} }}
        }});
        const candleSeries = chart.addCandlestickSeries({{ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' }});
        
        const data = JSON.parse('{chart_json}').map(item => ({{
            time: new Date(item.time).getTime() / 1000,
            open: item.open, high: item.high, low: item.low, close: item.close
        }}));
        candleSeries.setData(data);

        candleSeries.setMarkers([
            {{ time: {entry_time_unix}, position: 'belowBar', color: '#2196F3', shape: 'arrowUp', text: 'Entry @ {entry_price:.2f}' }}
        ]);
        chart.timeScale().fitContent();
    </script></body></html>
    """
    
    await page.set_content(html_content)
    await asyncio.sleep(0.5) # Give chart a moment to render
    
    screenshot_bytes = await page.screenshot()
    await browser_instance.close()
    await browser.stop()
    
    return screenshot_bytes

# --- Core Analysis Logic ---
def run_trade_analysis(df, tp, sl):
    """Calculates trade outcomes and returns a results DataFrame."""
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    results = []
    for index, row in df.iterrows():
        entry_price = row['open']
        entry_time = row['time']
        tp_price = entry_price * tp
        sl_price = entry_price * sl
        
        exit_info = {
            'Open_price': entry_price, 'W/L': "OPEN", 'open_datetime': entry_time,
            'end_datetime': None, 'Exit_price': None, 'TP_price': tp_price, 'SL_price': sl_price
        }

        future_df = df.iloc[index+1:]
        for _, next_row in future_df.iterrows():
            if next_row['high'] >= tp_price:
                exit_info.update({'W/L': "WIN", 'end_datetime': next_row['time'], 'Exit_price': tp_price})
                break
            if next_row['low'] <= sl_price:
                exit_info.update({'W/L': "LOSS", 'end_datetime': next_row['time'], 'Exit_price': sl_price})
                break
        
        results.append({**row.to_dict(), **exit_info})
        
    return pd.DataFrame(results)

# --- API Endpoints ---

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """
    Performs analysis, caches the data, and returns trade results without images.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        tp = float(request.form.get('tp', 1.006))
        sl = float(request.form.get('sl', 0.9994))
        
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        results_df = run_trade_analysis(df.copy(), tp, sl)
        
        # --- Caching Step ---
        analysis_id = str(uuid.uuid4())
        # Cache the original data for chart generation
        ANALYSIS_CACHE[analysis_id] = df 
        # Clean up old cache entries if necessary
        if len(ANALYSIS_CACHE) > MAX_CACHE_SIZE:
            oldest_key = next(iter(ANALYSIS_CACHE))
            del ANALYSIS_CACHE[oldest_key]
        
        # Prepare results for JSON response
        for col in ['time', 'open_datetime', 'end_datetime']:
            if col in results_df.columns:
                results_df[col] = results_df[col].astype(str).replace('NaT', None)
        
        return jsonify({
            "message": "Analysis successful.",
            "analysis_id": analysis_id, # Crucial for fetching charts later
            "trades": results_df.to_dict(orient='records')
        }), 200

    except Exception as e:
        logging.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/chart/<analysis_id>/<int:trade_index>', methods=['GET'])
def get_chart_endpoint(analysis_id, trade_index):
    """
    Generates and returns a single chart image on-demand.
    """
    if analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "Analysis ID not found or expired."}), 404
        
    try:
        # Retrieve the original data from the cache
        df = ANALYSIS_CACHE[analysis_id]
        
        # Get the specific trade details
        trade_details = df.iloc[trade_index].to_dict()
        trade_details['open_datetime'] = trade_details['time'] # Use original time for marker
        trade_details['Open_price'] = trade_details['open']
        
        # For the chart, show 60 candles leading up to the entry
        start_index = max(0, trade_index - 59)
        chart_data_slice = df.iloc[start_index : trade_index + 1]
        
        # Generate the chart image in memory
        image_bytes = asyncio.run(create_chart_screenshot(chart_data_slice, trade_details))
        
        # Return the image directly
        return send_file(io.BytesIO(image_bytes), mimetype='image/png')
        
    except Exception as e:
        logging.error(f"Chart generation error for {analysis_id}/{trade_index}: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate chart"}), 500

if __name__ == '__main__':
    # This block is for local development only.
    app.run(host='0.0.0.0', port=5555, debug=True)