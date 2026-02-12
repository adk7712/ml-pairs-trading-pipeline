# IMP DATES
START_DATE = "2015-01-01"
END_DATE   = "2024-01-01"
IN_SAMPLE_END = "2020-12-31"
OUT_OF_SAMPLE_START = "2021-01-01"

# TICKERS
STOCK_TICKERS = [
    # tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AVGO",
    "AMD", "ADBE", "CRM", "ORCL", "V", "MA",
    # finance
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    "AXP", "BLK", "SCHW", "SPGI", "PGR",
    # retail
    "WMT", "HD", "TGT", "PG", "KO",
    "PEP", "NKE", "MCD", "SBUX", "BABA", "DIS"
]

# ROLLING WINDOWS
ROLLING_BETA = 60
ZSCORE_WINDOW = 60

# SIGNAL THRESHOLDS
ENTRY_Z = 2.0
EXIT_Z = 0.5

# CAPITAL INFO
CAPITAL = 100_000
RISK_PER_TRADE = 0.01

# HALF LIFE FILTER INFO
MIN_HALF_LIFE = 20
MAX_HALF_LIFE = 250

# CLUSTERING
PCA_VARIANCE_THRESHOLD = 0.95

# COINTEGRATINO INFO
MIN_COINT_OBS = 30 # min observations required for coint test
PVAL_THRESHOLD = 0.05 # p-value threshold for cointegration test
MAX_HURST_EXPONENT = 0.45 # reject pairs with H >= 0.45

# STOP LOSS
STOP_LOSS_Z_THRESHOLD = 4.0 # z-score threshold for stop loss
STOP_EXIT_FACTOR = 3.0 # exit if position held for > (3 * half_life days)

# TRANSACTION FEES
COMMISSION_BPS = 1.5 # commission in basis points (1.5 bps = 0.015%)
