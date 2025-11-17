#!/bin/bash
# Automated documentation build with testing

set -e  # Exit on error

echo "========================================="
echo "Running Tests & Building Documentation"
echo "========================================="

# Step 1: Run all tests with coverage
echo ""
echo "Step 1: Running pytest with coverage..."
pytest --cov=core \
       --cov-report=html \
       --cov-report=term \
       --cov-report=json \
       --doctest-modules \
       -v

# Step 2: Extract coverage data to text for docs
echo ""
echo "Step 2: Extracting coverage data..."
coverage report --format=markdown > docs/coverage_report.md

# Step 3: Generate test results summary
echo ""
echo "Step 3: Generating test summary..."
pytest --tb=no --no-header -q > docs/test_results.txt 2>&1 || true

# Step 4: Build Sphinx documentation
echo ""
echo "Step 4: Building Sphinx documentation..."
cd docs
sphinx-build -b html . _build/html

# Step 5: Copy htmlcov into docs output
echo ""
echo "Step 5: Copying coverage HTML report..."
if [ -d "../htmlcov" ]; then
    cp -r ../htmlcov _build/html/
fi

echo ""
echo "========================================="
echo " Done! Open docs/_build/html/index.html"
echo "========================================="
EOF
