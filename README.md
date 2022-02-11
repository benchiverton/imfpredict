![Test using pytest](https://github.com/benchiverton/imfpredict/workflows/Test%20using%20pytest/badge.svg)![Lint using Flake8](https://github.com/benchiverton/imfpredict/workflows/Lint%20using%20Flake8/badge.svg)

# International Monetary Fund (IMF) Predict

Python project analysing & predicting currency performance using data from [International Monetary Fund (IMF)](https://www.imf.org/en/Home), taking inspiration from [The Alpha Achademy](https://www.alphavantage.co/academy/#lstm-for-finance).

## Getting Started

Checkout the repo and run the following commands in the base directory of the repo:

```bash
# install virtualenv
pip install virtualenv
# create your virtual environment at the base of the repo
python -m virtualenv .venv
# activate your virtual environment
.venv\Scripts\activate
# install tools used for the setup script
pip install -U pytest setuptools wheel build
# install project dependencies
python setup.py install
# note: you may need to restart your IDE after these steps in order for intellisense to work

# deactivate virtual environment
deactivate
```

## System Requirements

#### Installing Python (Windows)

**Python Version:** 3.8.x - 3.9.x

1. Download the installer from https://www.python.org/downloads/. **Ensure you install a supported version (see above).**
2. Select 'Customize instillation'
3. Click 'Next'
4. Check '*Add Python to environment variables*' & set the install location to `C:\Python`, then click 'Install'
5. Verify Python instillation by running `python -V`
6. Verify Pip instillation by running `pip -V`
