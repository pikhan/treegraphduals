# Makefile for Sphinx documentation with integrated testing

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PYTEST        = pytest
COVERAGE      = coverage

.PHONY: help clean html test coverage docs-with-tests

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html             Build HTML documentation only"
	@echo "  test             Run tests only"
	@echo "  coverage         Run tests with coverage only"
	@echo "  docs-with-tests  Run tests + coverage + build docs (RECOMMENDED)"
	@echo "  clean            Remove build artifacts"

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf ../htmlcov
	rm -rf ../.coverage
	rm -rf ../.pytest_cache
	rm -f coverage_report.md test_results.txt

test:
	@echo "Running pytest..."
	cd .. && $(PYTEST) -v

coverage:
	@echo "Running pytest with coverage..."
	cd .. && $(PYTEST) --cov=core --cov=timeseries --cov=visualizations --cov-report=html --cov-report=term -v

docs-with-tests:
	@echo "========================================="
	@echo "Running Tests & Building Documentation"
	@echo "========================================="
	@echo ""
	@echo "Step 1: Running all tests with coverage..."
	cd .. && $(PYTEST) --cov=core \
	                    --cov=timeseries \
	                    --cov=visualizations \
	                    --cov-report=html \
	                    --cov-report=term \
	                    --cov-report=json \
	                    --cov-report=markdown:docs/coverage_report.md \
	                    --doctest-modules \
	                    -v --tb=short | tee docs/test_results.txt
	@echo ""
	@echo "Step 2: Building Sphinx documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo ""
	@echo "Step 3: Copying coverage HTML..."
	@if [ -d "../htmlcov" ]; then \
		cp -r ../htmlcov $(BUILDDIR)/html/; \
		echo "Coverage report copied to _build/html/htmlcov/"; \
	fi
	@echo ""
	@echo "========================================="
	@echo "Done!"
	@echo "Open: $(BUILDDIR)/html/index.html"
	@echo "Coverage: $(BUILDDIR)/html/htmlcov/index.html"
	@echo "========================================="

html: clean
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Catch-all target
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)