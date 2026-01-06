#!/bin/bash
# Script to fix NumPy/matplotlib compatibility issue

echo "Fixing NumPy/matplotlib compatibility..."
echo "Option 1: Downgrading NumPy to < 2.0 (recommended)"
pip install "numpy<2"

echo ""
echo "If that doesn't work, try:"
echo "Option 2: Upgrade matplotlib"
echo "  pip install --upgrade matplotlib"
echo ""
echo "Option 3: Reinstall both"
echo "  pip uninstall numpy matplotlib"
echo "  pip install 'numpy<2' matplotlib"

