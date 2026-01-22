#!/bin/bash

# System check script for DL Accelerated MCX Simulation

echo "DL Accelerated MCX Simulation - System Check"
echo "========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "system_check.py" ]; then
    echo "Error: system_check.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Virtual environment activated"
    echo ""
fi

# Run system check
echo "Running system check..."
python system_check.py

echo ""
echo "System check completed!"