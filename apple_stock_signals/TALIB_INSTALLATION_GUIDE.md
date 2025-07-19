# TA-Lib Installation Guide

## Overview
TA-Lib (Technical Analysis Library) is required for advanced technical indicators in the trading system. This guide provides installation instructions for different operating systems.

## macOS Installation

### Method 1: Using Homebrew (Recommended)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install TA-Lib
brew install ta-lib

# Install Python wrapper
pip install TA-Lib
```

### Method 2: Manual Installation
```bash
# Download TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Build and install
./configure --prefix=/usr/local
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

## Linux Installation

### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential wget

# Download and install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

### CentOS/RHEL/Fedora
```bash
# Install dependencies
sudo yum install gcc gcc-c++ make wget

# Download and install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

## Windows Installation

### Method 1: Using Pre-built Wheels (Easiest)
1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Download the appropriate wheel file for your Python version and architecture
3. Install using pip:
   ```cmd
   pip install TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl
   ```

### Method 2: Using Anaconda
```cmd
conda install -c conda-forge ta-lib
```

## Alternative: Using the 'ta' Package

If TA-Lib installation is problematic, the trading system can work with the pure Python 'ta' package as a fallback:

```bash
pip install ta
```

Note: The 'ta' package provides similar functionality but may have slight differences in calculation methods.

## Verification

After installation, verify TA-Lib is working:

```python
python3 -c "import talib; print(talib.__version__)"
```

If successful, you should see the version number (e.g., "0.4.28").

## Troubleshooting

### Common Issues:

1. **"talib/_ta_lib.c:747:10: fatal error: ta-lib/ta_defs.h: No such file or directory"**
   - Solution: TA-Lib C library is not properly installed. Follow the installation steps for your OS.

2. **"ImportError: libta_lib.so.0: cannot open shared object file"**
   - Solution: Add library path:
     ```bash
     export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
     ```

3. **macOS M1/M2 Issues**
   - Use Homebrew with Apple Silicon support
   - Or install using conda-forge channel

### Fallback Mode

If TA-Lib cannot be installed, the trading system will automatically use the 'ta' package for technical indicators. While this provides most functionality, some advanced indicators may not be available.

## Contact Support

If you continue to experience issues:
1. Check the test output: `trade-test`
2. Review system health: `trade-health`
3. The system will work without TA-Lib but with reduced indicator options