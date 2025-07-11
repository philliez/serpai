#!/bin/bash

# -----------------------------------------------------------------------------
# Script to generate a chart for every trade in a CSV file using the API.
#
# Prerequisites:
# 1. The trade analysis API must be running.
# 2. `curl` and `jq` must be installed on your system.
# -----------------------------------------------------------------------------

# --- Configuration ---
# Set the base URL of your running API
API_URL="http://localhost:5555"

# Set the path to your input CSV file
CSV_FILE="/home/phill/serpai/trades.csv" # <--- IMPORTANT: CHANGE THIS PATH

# Set the directory where chart images will be saved
OUTPUT_DIR="trade_charts_output"

# --- Script Start ---

# 1. Validate that the input CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: Input CSV file not found at '$CSV_FILE'"
    exit 1
fi

# 2. Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Charts will be saved in the '$OUTPUT_DIR' directory."
echo "-----------------------------------"

# 3. Step 1: Call the /analyze endpoint to get the analysis_id
echo "[INFO] Step 1: Uploading CSV to get the analysis ID..."
ANALYSIS_RESPONSE=$(curl --silent -X POST \
  -F "file=@$CSV_FILE" \
  -F "tp=1.006" \
  -F "sl=0.9994" \
  "$API_URL/analyze")

# 4. Parse the JSON response to extract the analysis_id and total trades
ANALYSIS_ID=$(echo "$ANALYSIS_RESPONSE" | jq -r '.analysis_id')
TOTAL_TRADES=$(echo "$ANALYSIS_RESPONSE" | jq '.trades | length')
ERROR_MSG=$(echo "$ANALYSIS_RESPONSE" | jq -r '.error')

# 5. Validate the response from the /analyze endpoint
if [ "$ANALYSIS_ID" == "null" ] || [ -z "$ANALYSIS_ID" ]; then
    echo "Error: Failed to get a valid analysis ID from the API."
    echo "API Response: $ERROR_MSG"
    exit 1
fi

if [ "$TOTAL_TRADES" -eq 0 ]; then
    echo "Warning: Analysis was successful, but no trades were found to process."
    exit 0
fi

echo "[SUCCESS] Got Analysis ID: $ANALYSIS_ID"
echo "[INFO] Found $TOTAL_TRADES trades to process."
echo "-----------------------------------"

# 6. Step 2: Loop through each trade and download the chart
echo "[INFO] Step 2: Starting chart generation for each trade..."

for i in $(seq 0 $((TOTAL_TRADES - 1)))
do
    TRADE_INDEX=$i
    CHART_URL="$API_URL/chart/$ANALYSIS_ID/$TRADE_INDEX"
    OUTPUT_FILE="$OUTPUT_DIR/chart_trade_${TRADE_INDEX}.png"
    
    echo "  -> Downloading chart for trade index $TRADE_INDEX..."
    
    # Use curl to download the image for the current trade index
    curl --silent -o "$OUTPUT_FILE" "$CHART_URL"
    
    # Check if the file was created successfully
    if [ -s "$OUTPUT_FILE" ]; then
        echo "     [SUCCESS] Saved to $OUTPUT_FILE"
    else
        echo "     [ERROR] Failed to download chart for trade index $TRADE_INDEX"
    fi
done

echo "-----------------------------------"
echo "[COMPLETE] All chart generation tasks are finished."