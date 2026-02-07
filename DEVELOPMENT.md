# Development Setup Guide

## Quick Start

1. **Clone or navigate to the project directory**
   ```bash
   cd peak_analyzer
   ```

2. **Install in development mode**
   ```bash
   # Using uv (recommended)
   uv add --dev ".[dev,examples]"
   
   # Or using pip
   pip install -e ".[dev,examples]"
   ```

3. **Run the demonstration**
   ```bash
   python main.py
   ```

4. **Run tests**
   ```bash
   pytest
   ```

## Detailed Setup

### Prerequisites

- Python 3.10 or higher
- uv (recommended) or pip for package management

### Installation Options

#### Option 1: Development Installation (Recommended)
```bash
# Install package in editable mode with all dependencies
uv add --dev ".[dev,examples]"

# Or with pip
pip install -e ".[dev,examples]"
```

#### Option 2: Basic Installation
```bash
# Install only core dependencies
uv add .

# Or with pip  
pip install .
```

#### Option 3: User Installation
```bash
# Install from PyPI (when published)
pip install peak-analyzer
```

### Development Environment Setup

#### 1. Code Quality Tools
```bash
# Format code with Black
black peak_analyzer/ tests/ examples/

# Lint code with flake8
flake8 peak_analyzer/ tests/ examples/

# Run all quality checks
black --check peak_analyzer/ && flake8 peak_analyzer/
```

#### 2. Testing Setup
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=peak_analyzer

# Run only fast tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_peak_analyzer.py

# Run with verbose output
pytest -v
```

#### 3. Running Examples
```bash
# Basic usage example
python examples/basic_usage.py

# Advanced features example  
python examples/advanced_usage.py

# Main demonstration
python main.py
```

## Project Structure for Development

```
peak_analyzer/
├── peak_analyzer/          # Source code
├── tests/                  # Test files
├── examples/               # Usage examples  
├── main.py                 # Demo script
├── pyproject.toml         # Project configuration
├── ARCHITECTURE.md        # Architecture documentation
└── README.md              # Main documentation
```

## Development Workflow

### 1. Adding New Features

```bash
# Create feature branch
git checkout -b feature/new-calculator

# Add your code to appropriate module
# e.g., peak_analyzer/features/new_calculator.py

# Add corresponding tests
# e.g., tests/test_new_calculator.py

# Run tests
pytest tests/test_new_calculator.py

# Format and lint
black peak_analyzer/ tests/
flake8 peak_analyzer/ tests/

# Commit changes
git commit -m "Add new calculator feature"
```

### 2. Testing Changes

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=peak_analyzer --cov-report=html

# Run integration tests
pytest -m integration

# Test specific functionality
pytest tests/test_peak_analyzer.py::TestPeakAnalyzer::test_feature_name
```

### 3. Documentation Updates

```bash
# Update README.md for user-facing changes
# Update ARCHITECTURE.md for design changes  
# Add examples to examples/ directory
# Update docstrings in code
```

## Common Development Tasks

### Adding a New Feature Calculator

1. Create the calculator module:
   ```python
   # peak_analyzer/features/my_calculator.py
   class MyCalculator:
       def calculate(self, peak_region):
           # Implementation here
           return result
   ```

2. Add to features `__init__.py`:
   ```python
   from .my_calculator import MyCalculator
   __all__.append("MyCalculator")
   ```

3. Integrate with LazyDataFrame:
   ```python
   # In peak_analyzer/core/lazy_dataframe.py
   elif feature_name == "my_feature":
       calculator = MyCalculator()
       self._feature_cache[feature_name] = np.array([
           calculator.calculate(region) for region in self.peak_regions
       ])
   ```

4. Add tests:
   ```python
   # tests/test_my_calculator.py
   def test_my_calculator():
       # Test implementation
   ```

### Adding a New Detection Strategy

1. Create strategy module:
   ```python
   # peak_analyzer/strategies/my_strategy.py
   class MyStrategy:
       def process_peaks(self, peak_regions, prominence_calculator):
           # Implementation here
           return processed_regions
   ```

2. Register in PeakAnalyzer:
   ```python
   # In peak_analyzer/core/peak_detector.py
   def _get_strategy(self, strategy_name):
       strategies = {
           "my_strategy": MyStrategy(),
           # ... existing strategies
       }
   ```

### Debugging Tips

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your peak detection code here
```

#### Profile Performance
```python
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

# Your peak detection code here

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(10)
```

#### Memory Usage
```python
import tracemalloc

tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you installed in development mode with `-e` flag
2. **Missing Dependencies**: Run `uv add` or `pip install -e ".[dev,examples]"`
3. **Test Failures**: Check if all required packages are installed
4. **Performance Issues**: Use appropriate strategy for your data type

### Getting Help

1. Check the examples in `examples/` directory
2. Review tests in `tests/` for usage patterns
3. Read the architecture documentation in `ARCHITECTURE.md`
4. Open an issue on the project repository

## Contributing Guidelines

1. **Code Style**: Use Black formatter and follow PEP 8
2. **Testing**: Add tests for all new features
3. **Documentation**: Update relevant documentation
4. **Performance**: Consider lazy evaluation and memory efficiency  
5. **Compatibility**: Maintain N-dimensional generalization