# stock-singlas-public

Public market data from Yahoo Finance, refreshed by GitHub Actions on US weekdays
(about 07:45 and 16:20 New York time). Contains market data only: no positions, costs or account details.

Read it from anywhere without credentials:

    git clone --depth 1 https://github.com/singalrohit/stock-singlas-public
    # or a single file
    curl -s https://raw.githubusercontent.com/singalrohit/stock-singlas-public/main/data/latest.json

Files in `data/`: status.json (run health), latest.json, metrics.json, history/<T>.csv,
events.json, earnings_reactions.json, fomc_reactions.json, implied_moves.json,
fundamentals.json, screen.json, regime.json, snapshots/.
