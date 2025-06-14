#!/bin/bash

# Improved build script for Zenoh D435i Subscriber Python package
# This version ensures reliable builds by detecting the correct Python environment

set -e

echo "🚀 Building Zenoh D435i Subscriber Python package..."

# Detect current Python interpreter
PYTHON_PATH=$(which python)
PYTHON_VERSION=$(python --version)
echo "🐍 Using Python: $PYTHON_PATH ($PYTHON_VERSION)"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "📦 Installing maturin..."
    pip install maturin
fi

# Check if numpy is installed
if ! python -c "import numpy" &> /dev/null; then
    echo "📦 Installing numpy..."
    pip install numpy
fi

# Clean previous builds to ensure a fresh start
echo "🧹 Cleaning previous builds..."
cargo clean

# Build the wheel
echo "🔧 Building wheel..."
if [ "$1" = "release" ]; then
    echo "🚀 Building in release mode..."
    maturin build --release
else
    echo "🔧 Building in development mode..."
    maturin build
fi

# Find the built wheel
WHEEL_PATH=$(find target/wheels -name "zenoh_d435i_subscriber-*.whl" | head -1)

if [ -z "$WHEEL_PATH" ]; then
    echo "❌ Error: No wheel found!"
    exit 1
fi

echo "📦 Found wheel: $WHEEL_PATH"

# Install the wheel
echo "📥 Installing wheel..."
pip install "$WHEEL_PATH" --force-reinstall

# Verify installation
echo "✅ Verifying installation..."
if python -c "import zenoh_d435i_subscriber; print('✅ Import successful!')"; then
    echo "🎉 Build and installation completed successfully!"
    echo ""
    echo "📋 Available classes:"
    python -c "import zenoh_d435i_subscriber; print([name for name in dir(zenoh_d435i_subscriber) if not name.startswith('_')])"
    echo ""
    echo "🚀 You can now run: python python/example.py"
else
    echo "❌ Import failed!"
    exit 1
fi 