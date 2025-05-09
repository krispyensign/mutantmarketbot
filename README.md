# MutantMarketBot

MutantMarketBot is a bot to perform signal generation and backtesting using WMA on different charts.

## NOTICE

This application is provided "AS IS" for educational purposes only. It does not replace perfectly valid trading
solutions already available. DO NOT USE WITHOUT FIRST TESTING YOURSELF.

## Quickstart steps

```shell
# install ta-lib
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb
sudo apt install -y ./ta-lib_0.6.4_amd64.deb

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the deps
uv pip install '.'

# or
uv pip install '.[dev]'

# setup Oanda tokens
export OANDA_TOKEN=$YOUR_OANDA_TOKEN
export OANDA_ACCOUNT_ID=$YOUR_OANDA_ACCOUNT_ID

# bot mode
python src/main.py bot ./example_configs/config.yaml

# backtest mode 
python src/main.py backtest ./example_configs/config.yaml

# solver mode
python src/main.py solve ./example_configs/config.yaml
```
