#!/usr/bin/env bash
set -e

echo "Bootstrapping ONNX9000 Polyglot Monorepo..."

# Check for required tools
if ! command -v pnpm &> /dev/null; then
    echo "pnpm could not be found. Please install it: npm install -g pnpm"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Installing JS dependencies..."
pnpm install

echo "Creating Python virtual environment..."
uv venv

echo "Syncing Python workspace..."
# Install Python packages in editable mode
uv pip install -e packages/python/onnx9000-core \
               -e packages/python/onnx9000-backend-native \
               -e packages/python/onnx9000-optimizer \
               -e packages/python/onnx9000-frontend \
               -e packages/python/onnx9000-toolkit || echo "Warning: Some Python packages may not exist yet."

echo "Bootstrap complete!"