#!/bin/bash

# Build script for Zenoh D435i Subscriber Python package

set -e

echo "Building Zenoh D435i Subscriber Python package..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Check if numpy is installed
if ! python -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    pip install numpy
fi

# Build the package
echo "Building package..."
if [ "$1" = "release" ]; then
    echo "Building in release mode..."
    maturin develop --release
else
    echo "Building in development mode..."
    maturin develop
fi

echo "Build complete!"
echo "You can now run: python python/example.py" 