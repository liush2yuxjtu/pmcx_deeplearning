#!/bin/bash

# Run all validations for DL Accelerated MCX Simulation

echo "DL Accelerated MCX Simulation - Complete Validation"
echo "=================================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "final_verification.py" ]; then
    echo "Error: final_verification.py not found in current directory"
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
echo "1. Running system check..."
./system_check.sh
echo ""

# Run final verification
echo "2. Running final verification..."
python final_verification.py
echo ""

# Run unit tests
echo "3. Running unit tests..."
python -m pytest tests/ -v
echo ""

# Run example
echo "4. Running example..."
python examples/example_usage.py
echo ""

echo "Complete validation finished!"