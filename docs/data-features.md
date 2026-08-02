# Data & Features

Everything the portfolio pipeline consumes is aligned onto a single
`(asset, trading_day)` grid before it reaches the model, so a slice
`[:, t-T+1:t+1]` is always a synchronized lookback window across every branch
and every asset.

## Universe and calendar

- **Universe** (`tyche/portfolio/data/universe.py`) — resolved from the data,
  not hard-coded: symbols carrying **both** prices and news, optionally
  restricted to those with a complete history, ranked by median dollar volume
  and capped at `TYCHE_PORTFOLIO_UNIVERSE_SIZE`. The cap is a hardware
  constraint — the model emits an `N x N` covariance and factorizes it every
  forward pass, so cost grows as `N^3`. Symbols are canonicalized to their bare
  upper-cased form (`NASDAQ:AAON` → `AAON`), which is how the price and news
  feeds join.
- **Calendar** (`tyche/portfolio/data/calendar.py`) — the trading-day index is
  derived once from the intersection of dates every universe asset shares, so
  a sample at day `t` is guaranteed to have every asset present.

## Loaders

`tyche/portfolio/data/loaders.py` reads the two raw parquet sources (daily
OHLCV, news sentiment), normalizes symbols, coerces timestamps, and returns tidy
long frames. The price source is the merged long-format file produced by
`scripts/merge_rl2k_ohlcv.sh`. Loaders carry no feature
logic — that lives entirely in the `features_*` modules below.

## Feature branches

Two independent branches, each strictly causal (a feature at day `d` only
uses information available at or before `d`'s close):

- **Daily** (`features/daily.py`) — normalized OHLCV-derived features (returns,
  ranges, ratios, rolling volatility/momentum/RSI/ATR/z-score), one row per
  `(asset, date)`. Raw prices are never fed directly to the model.
- **News** (`features/news.py`) — exact-window, centroid-representative story
  sentiment. For decision day `t`, the window is strictly lagged: effective-news
  dates `t−30` through `t−1`; articles mapped to `t` are excluded. Articles in
  that window are clustered into stories via a cosine-similarity graph over local
  text embeddings (GPU-accelerated when available); each story is represented by
  its centroid-closest article. Daily features are the mean representative
  sentiment and the log count of unique stories in the window. Days with no news
  are zero, never forward-filled.

## Windowing and splitting

`tyche/portfolio/data/windows.py` builds windowed cross-sectional samples: a
sample at decision day `t` carries the synchronized `T`-day lookback across
all branches/assets, with a target of the forward `H`-day log return
(`t → t+H`). The split (`tyche/portfolio/config.SplitConfig`) is strictly
chronological on the trading-day index — no shuffling — with an `embargo`
(≥ `H` trading days) dropped after each split boundary so no sample's target
window overlaps the next split.

## Standardization

`tyche/portfolio/data/preprocessing.py` fits per-feature mean/std on the
training split only, then applies the same transform unchanged to
validation/test. Remaining NaNs (warm-up, undefined ratios, missing bars) are
zero-filled *after* standardization; the news `no_news` binary flag passes
through untouched.

## Assembly

`tyche/portfolio/data/assemble.py` (`AlignedData`) is the point where the
two standardized branches, the universe, the calendar, and the forward-
return target all come together into the dense arrays the model and backtest
consume.
