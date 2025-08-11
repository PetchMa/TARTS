#!/bin/bash

# Test script to build the TARTS conda package locally
# This script will help verify that your conda recipe works correctly

set -e  # Exit on any error

echo "=== TARTS Conda Package Build Test ==="
echo "Current directory: $(pwd)"
echo ""

# Check if we're in the right directory
if [ ! -f "meta.yaml" ]; then
    echo "Error: meta.yaml not found. Please run this script from the conda-recipe directory."
    exit 1
fi

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo "Installing conda-build..."
    conda install -c conda-forge conda-build -y
fi

# Clean any previous builds
echo "Cleaning previous builds..."
conda build purge

# Build the package
echo "Building TARTS conda package..."
echo "This may take several minutes..."

# Build with verbose output and show the package location
conda build . --output-folder ./build_output --verbose

echo ""
echo "=== Build completed successfully! ==="
echo "Package files are in: ./build_output"
echo ""

# List the built packages
echo "Built packages:"
ls -la ./build_output/*/TARTS-*.tar.bz2 2>/dev/null || echo "No .tar.bz2 files found"
ls -la ./build_output/*/TARTS-*.conda 2>/dev/null || echo "No .conda files found"

echo ""
echo "To install the built package locally:"
echo "conda install --use-local tarts"
echo ""
echo "To test the package:"
echo "conda create -n test_tarts python=3.9 -y"
echo "conda activate test_tarts"
echo "conda install --use-local tarts -y"
echo "python -c 'import TARTS; print(\"Package imported successfully!\")'"
