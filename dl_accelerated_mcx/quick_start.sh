#!/bin/bash

# DL Accelerated MCX Simulation - Quick Start Script

echo "🚀 DL Accelerated MCX Simulation - Quick Start"
echo "==========================================="
echo ""

# Function to show usage
show_usage() {
    echo "Usage:"
    echo "  ./quick_start.sh [MODE] [DATA_PATH]"
    echo ""
    echo "Modes:"
    echo "  simple          Fast simple mode (~1 minute)"
    echo "  full            High quality full mode (>30 minutes)"
    echo "  dl              DL accelerated mode (<2 minutes)"
    echo "  gui             Launch graphical interface"
    echo "  web             Launch web interface"
    echo "  check           Run system check"
    echo "  validate        Run validation tests"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh simple /path/to/data.nii.gz"
    echo "  ./quick_start.sh full /path/to/data.nii.gz"
    echo "  ./quick_start.sh dl /path/to/data.nii.gz"
    echo "  ./quick_start.sh gui"
    echo "  ./quick_start.sh web"
    echo "  ./quick_start.sh check"
    echo "  ./quick_start.sh validate"
    echo ""
}

# Check if arguments provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Parse arguments
MODE=$1
DATA_PATH=$2

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo ""
fi

# Run based on mode
case $MODE in
    "simple")
        if [ -z "$DATA_PATH" ]; then
            echo "Error: DATA_PATH is required for simple mode"
            show_usage
            exit 1
        fi
        echo "⚡ Running simple mode simulation..."
        echo "📂 Data path: $DATA_PATH"
        python cli.py --mode simple --data-path "$DATA_PATH" --output-dir ./results_simple
        ;;
        
    "full")
        if [ -z "$DATA_PATH" ]; then
            echo "Error: DATA_PATH is required for full mode"
            show_usage
            exit 1
        fi
        echo "🔬 Running full mode simulation..."
        echo "📂 Data path: $DATA_PATH"
        python cli.py --mode full --data-path "$DATA_PATH" --output-dir ./results_full
        ;;
        
    "dl")
        if [ -z "$DATA_PATH" ]; then
            echo "Error: DATA_PATH is required for DL accelerated mode"
            show_usage
            exit 1
        fi
        echo "🚀 Running DL accelerated simulation..."
        echo "📂 Data path: $DATA_PATH"
        python cli.py --mode dl_accelerated --data-path "$DATA_PATH" --output-dir ./results_dl
        ;;
        
    "gui")
        echo "🖥️ Launching graphical interface..."
        python gui.py
        ;;
        
    "web")
        echo "🌐 Launching web interface..."
        python web.py
        ;;
        
    "check")
        echo "🔍 Running system check..."
        ./check.sh
        ;;
        
    "validate")
        echo "🧪 Running validation tests..."
        python final_verification.py
        ;;
        
    *)
        echo "Error: Unknown mode '$MODE'"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "✅ Operation completed!"