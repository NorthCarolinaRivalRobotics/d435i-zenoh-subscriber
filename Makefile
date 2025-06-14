.PHONY: build build-release clean test install dev-deps help

# Default target
help:
	@echo "🚀 Zenoh D435i Subscriber Build Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build         - Build in development mode"
	@echo "  make build-release - Build in release mode"
	@echo "  make test          - Test the installation"
	@echo "  make example       - Run the example"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make dev-deps      - Install development dependencies"
	@echo "  make install       - Quick install (build + install)"
	@echo ""

# Build in development mode
build:
	@echo "🔧 Building in development mode..."
	@./build.sh

# Build in release mode  
build-release:
	@echo "🚀 Building in release mode..."
	@./build.sh release

# Quick install (most common use case)
install: build
	@echo "✅ Installation complete!"

# Test the installation
test:
	@echo "🧪 Testing installation..."
	@python -c "import zenoh_d435i_subscriber; print('✅ Import successful!')"
	@python -c "import zenoh_d435i_subscriber as zd; print('📋 Classes:', [n for n in dir(zd) if not n.startswith('_')])"

# Run the example
example: test
	@echo "🎯 Running example..."
	@python python/example.py

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -rf target/wheels
	@rm -rf *.png
	@echo "✅ Clean complete!"

# Install development dependencies
dev-deps:
	@echo "📦 Installing development dependencies..."
	@pip install maturin pytest opencv-python pillow matplotlib
	@echo "✅ Development dependencies installed!" 