#!/bin/bash

# DL Accelerated MCX Simulation Launcher

echo "🚀 DL Accelerated MCX Simulation Launcher"
echo "========================================"
echo ""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode MODE          Simulation mode (simple, full, dl_accelerated)"
    echo "  --data-path PATH     Path to NIfTI tissue segmentation file"
    echo "  --output-dir DIR     Output directory (default: ./results)"
    echo "  --gui                Launch GUI interface"
    echo "  --web                Launch web interface"
    echo "  --check              Run system check"
    echo "  --validate           Run validation"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mode simple --data-path /path/to/data.nii.gz"
    echo "  $0 --mode full --data-path /path/to/data.nii.gz"
    echo "  $0 --mode dl_accelerated --data-path /path/to/data.nii.gz"
    echo "  $0 --gui"
    echo "  $0 --web"
    echo "  $0 --check"
    echo "  $0 --validate"
    echo ""
}

# Parse command line arguments
MODE=""
DATA_PATH=""
OUTPUT_DIR="./results"
LAUNCH_GUI=false
LAUNCH_WEB=false
RUN_CHECK=false
RUN_VALIDATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gui)
            LAUNCH_GUI=true
            shift
            ;;
        --web)
            LAUNCH_WEB=true
            shift
            ;;
        --check)
            RUN_CHECK=true
            shift
            ;;
        --validate)
            RUN_VALIDATE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo ""
fi

# Run system check if requested
if [ "$RUN_CHECK" = true ]; then
    echo "🔍 Running system check..."
    ./check.sh
    exit 0
fi

# Run validation if requested
if [ "$RUN_VALIDATE" = true ]; then
    echo "🧪 Running validation..."
    python final_verification.py
    exit 0
fi

# Launch GUI if requested
if [ "$LAUNCH_GUI" = true ]; then
    echo "🖥️ Launching GUI interface..."
    python gui.py
    exit 0
fi

# Launch web interface if requested
if [ "$LAUNCH_WEB" = true ]; then
    echo "🌐 Launching web interface..."
    python web.py
    exit 0
fi

# Run simulation if mode is specified
if [ -n "$MODE" ]; then
    if [ -z "$DATA_PATH" ]; then
        echo "❌ Error: --data-path is required when specifying a mode"
        usage
        exit 1
    fi
    
    echo "⚡ Running simulation in $MODE mode..."
    echo "📂 Data path: $DATA_PATH"
    echo "📁 Output directory: $OUTPUT_DIR"
    echo ""
    
    python cli.py \
        --mode "$MODE" \
        --data-path "$DATA_PATH" \
        --output-dir "$OUTPUT_DIR"
        
    exit 0
fi

# If no arguments provided, show usage
usage
exit 1