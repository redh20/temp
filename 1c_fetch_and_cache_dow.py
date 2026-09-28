# -*- coding: utf-8 -*-
"""
1c_fetch_and_cache_dow.py
=========================
DOW JONES INDUSTRIAL AVERAGE version of step 1 (S&P 500: 1_fetch_and_cache_data.py,
Nasdaq-100: 1b_fetch_and_cache_ndx100.py).
Run this FIRST, whenever you want fresh data (once a day is
plenty - prices don't change intraday in a way that matters for a
monthly-rebalanced strategy). It downloads every ticker needed for
RUN_YEARS, checks it for bad data, looks up sectors, and saves
EVERYTHING into one cache file. The other three scripts below just load
that cache file - they never touch the network themselves, so they run
in seconds instead of minutes, and (this is the important part) they
are GUARANTEED to use the exact same data as each other, because it's
literally the same file. That's what fixes the "two scripts gave
different totals" problem from before: there was never a real bug in
the comparison logic - it was two separate downloads of live data,
taken at slightly different moments, being compared as if they had to
match. Now there's only ever one download to compare against.

WHAT THIS SAVES: a single pickle file containing the price DataFrame,
the dow_by_year universe (filtered to RUN_YEARS, stored under the
cache key 'sp500_by_year' so the downstream scripts can read it unchanged), the sector lookup,
and a small metadata block (when it was fetched, which tickers came
back empty) - see CACHE_FILE below. The three downstream scripts each
print that same metadata when they load it, so you always know exactly
how fresh the data you're working with is.

Run this in Google Colab (needs yfinance's network access).
"""
import os
import json
import time
import pickle
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _EASTERN_TZ = ZoneInfo("America/New_York")
except Exception:
    _EASTERN_TZ = None  # tzdata unavailable - now_eastern() falls back to naive local time


def now_eastern():
    """Current time in US Eastern (handles EST/EDT automatically) - falls
    back to naive local time if the zoneinfo/tzdata package isn't
    available in this environment."""
    if _EASTERN_TZ is not None:
        return datetime.now(_EASTERN_TZ)
    return datetime.now()

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RUN_YEARS = list(range(2000, 2027))  # ONE LINE controls the whole pipeline's
                                    # span - all three downstream scripts
                                    # inherit this through the cache file.
                                    # e.g. list(range(2015, 2027)) for 2015
                                    # onward, or list(dow_by_year.keys())
                                    # for the full 2000-2026 history.
BENCHMARK_TICKER = 'DIA'
DATA_START_DATE_FALLBACK = '1999-01-01'  # only used if RUN_YEARS is somehow
                                    # empty; normally overridden below to
                                    # f'{min(RUN_YEARS)-1}-01-01' so a
                                    # narrow RUN_YEARS downloads a lot less.
SECTOR_CACHE_FILE = 'dow_sector_cache.json'  # caches each ticker's GICS
                                                # sector (from yfinance's
                                                # .info) so re-runs don't
                                                # re-fetch it every time.
EXCLUDE_INCOMPLETE_CURRENT_MONTH = False  # False = include the current,
                                          # still-open month using the
                                          # latest available price. True =
                                          # drop it, so every run before
                                          # month-end is IDENTICAL, but the
                                          # current month's numbers are
                                          # frozen until it closes.
EXCLUDE_PRICE_ANOMALIES = True     # yfinance occasionally returns a badly
                                    # corrupted price series for a specific
                                    # ticker (a recycled ticker symbol
                                    # picking up a different company's
                                    # price level, a botched split
                                    # adjustment). True = detect and drop
                                    # any such ticker's ENTIRE history
                                    # before it can distort anything
                                    # downstream, and print exactly which
                                    # ticker/date/prices triggered it.
MAX_MONTHLY_PRICE_RATIO = 10.0     # a real stock very rarely multiplies
                                    # (or divides) in price by more than
                                    # this within a single calendar month;
                                    # a bigger move is almost always bad
                                    # data, not a real market move.
RESTRICT_TO_MEMBERSHIP_WINDOW = True  # yfinance knows tickers by their
                                    # CURRENT owner, so an old index ticker
                                    # that was later reused by a different
                                    # company (SE->Sea Ltd, S->SentinelOne,
                                    # SUN->Sunoco LP, POM, STI, MI...) comes
                                    # back with that other company's prices.
                                    # True = keep each ticker's prices only
                                    # from 1 year before its first index year to
                                    # 1 year after its last one, so reused-
                                    # ticker data outside that window is
                                    # blanked, and the anomaly check below
                                    # only looks inside the window (a real,
                                    # out-of-window move like GME's Jan-2021
                                    # squeeze no longer gets GME's 2008-2015
                                    # index history thrown away).
KNOWN_REUSED_TICKERS = {            # reused tickers whose NEW owner's price
                                    # history overlaps the OLD company's index
                                    # years - the window above can't separate
                                    # them, so their data is dropped outright.
    'GM': 'General Motors (old GM, Dow until 2009, bankrupt 2009) - ticker taken by '
          'the new GM at its Nov 2010 IPO',
}
CACHE_FILE_NAME = 'momentum_data_cache_dow.pkl'  # deliberately DIFFERENT from the
                                    # S&P 500 cache so the two never overwrite
                                    # each other. To run scripts 2-4 on this
                                    # data, set their CACHE_FILE_NAME to this.

# This script needs real yfinance network access, so it only makes sense
# to run in Colab - and when it IS running in Colab, mount Google Drive
# and save there so the cache (and later, each script's own export)
# survives after the runtime disconnects.
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    SAVE_DIR = '/content/drive/MyDrive/momentum_strategy_data'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Google Drive mounted - saving to: {SAVE_DIR}")
except ImportError:
    SAVE_DIR = '.'
    print(f"Not running in Colab (google.colab not available) - saving locally to: {os.path.abspath(SAVE_DIR)}")
except Exception as e:
    SAVE_DIR = '.'
    print(f"Google Drive mount failed ({e}) - falling back to local save: {os.path.abspath(SAVE_DIR)}")

CACHE_FILE = os.path.join(SAVE_DIR, CACHE_FILE_NAME)

# ----------------------------------------------------------------------
# The DOW JONES INDUSTRIAL AVERAGE membership, one snapshot per calendar
# year-end (point in time), 2000-2026. RECONSTRUCTED from the index's
# published change history (Wikipedia: "Historical components of the Dow
# Jones Industrial Average") - there is no reliable point-in-time Dow list
# on GitHub (the pokedo0/index-constituents history has today's members,
# e.g. AMZN and NVDA, back-filled into the 1980s-2000s). The 2026 list
# matches that repo's current list. Changes applied to the Nov-1999 30:
#   2004-04  +AIG +PFE +VZ           -EK -IP -AT&T(old)
#   2008-02  +BAC +CVX               -HON -MO
#   2008-09  +KFT                    -AIG
#   2009-06  +CSCO +TRV              -C -GM
#   2012-09  +UNH                    -KFT
#   2013-09  +GS +NKE +V             -AA -BAC -HPQ
#   2015-03  +AAPL                   -T
#   2017-09  DuPont -> DowDuPont (same yfinance ticker, DD)
#   2018-06  +WBA                    -GE
#   2019-04  +DOW                    -DD (DowDuPont)
#   2020-08  +AMGN +HON +CRM         -XOM -PFE -RTX (Raytheon Tech.)
#   2024-02  +AMZN                   -WBA
#   2024-11  +NVDA +SHW              -INTC -DOW
# Tickers are the ones yfinance knows the same company by today:
#   AA->HWM (Alcoa Inc became Arconic, now Howmet)  SBC->T (SBC Communications
#   became today's AT&T)  UTX->RTX  HWP->HPQ  KFT->MDLZ  DWDP->DD.
# The ORIGINAL AT&T Corp (in the Dow until 2004) shares the ticker T with
# today's AT&T, so it is listed as 'T_OLD' (no Yahoo data - shows as missing).
# Expect missing/partial data for: GM (old GM, bankrupt 2009), EK (Kodak),
# T_OLD, WBA (acquired 2025), and possibly DD before 2017.
# The membership changes above are from memory of the public record, not
# checked here against a primary source for every date.
# ----------------------------------------------------------------------
dow_by_year = {
    2000: list(set([  # snapshot as of 2000-12-31, 30 tickers
        'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'EK', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'IP', 'JNJ', 'JPM', 'KO', 'MCD',
        'MMM', 'MO', 'MRK', 'MSFT', 'PG', 'RTX', 'T', 'T_OLD', 'WMT', 'XOM',
    ])),
    2001: list(set([  # snapshot as of 2001-12-31, 30 tickers
        'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'EK', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'IP', 'JNJ', 'JPM', 'KO', 'MCD',
        'MMM', 'MO', 'MRK', 'MSFT', 'PG', 'RTX', 'T', 'T_OLD', 'WMT', 'XOM',
    ])),
    2002: list(set([  # snapshot as of 2002-12-31, 30 tickers
        'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'EK', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'IP', 'JNJ', 'JPM', 'KO', 'MCD',
        'MMM', 'MO', 'MRK', 'MSFT', 'PG', 'RTX', 'T', 'T_OLD', 'WMT', 'XOM',
    ])),
    2003: list(set([  # snapshot as of 2003-12-31, 30 tickers
        'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'EK', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'IP', 'JNJ', 'JPM', 'KO', 'MCD',
        'MMM', 'MO', 'MRK', 'MSFT', 'PG', 'RTX', 'T', 'T_OLD', 'WMT', 'XOM',
    ])),
    2004: list(set([  # snapshot as of 2004-12-31, 30 tickers
        'AIG', 'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MO', 'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'VZ', 'WMT', 'XOM',
    ])),
    2005: list(set([  # snapshot as of 2005-12-31, 30 tickers
        'AIG', 'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MO', 'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'VZ', 'WMT', 'XOM',
    ])),
    2006: list(set([  # snapshot as of 2006-12-31, 30 tickers
        'AIG', 'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MO', 'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'VZ', 'WMT', 'XOM',
    ])),
    2007: list(set([  # snapshot as of 2007-12-31, 30 tickers
        'AIG', 'AXP', 'BA', 'C', 'CAT', 'DD', 'DIS', 'GE', 'GM', 'HD',
        'HON', 'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MO', 'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'VZ', 'WMT', 'XOM',
    ])),
    2008: list(set([  # snapshot as of 2008-12-31, 30 tickers
        'AXP', 'BA', 'BAC', 'C', 'CAT', 'CVX', 'DD', 'DIS', 'GE', 'GM',
        'HD', 'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MDLZ',
        'MMM', 'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'VZ', 'WMT', 'XOM',
    ])),
    2009: list(set([  # snapshot as of 2009-12-31, 30 tickers
        'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'HD',
        'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MDLZ', 'MMM',
        'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'TRV', 'VZ', 'WMT', 'XOM',
    ])),
    2010: list(set([  # snapshot as of 2010-12-31, 30 tickers
        'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'HD',
        'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MDLZ', 'MMM',
        'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'TRV', 'VZ', 'WMT', 'XOM',
    ])),
    2011: list(set([  # snapshot as of 2011-12-31, 30 tickers
        'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'HD',
        'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MDLZ', 'MMM',
        'MRK', 'MSFT', 'PFE', 'PG', 'RTX', 'T', 'TRV', 'VZ', 'WMT', 'XOM',
    ])),
    2012: list(set([  # snapshot as of 2012-12-31, 30 tickers
        'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'HD',
        'HPQ', 'HWM', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
        'MSFT', 'PFE', 'PG', 'RTX', 'T', 'TRV', 'UNH', 'VZ', 'WMT', 'XOM',
    ])),
    2013: list(set([  # snapshot as of 2013-12-31, 30 tickers
        'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
        'PFE', 'PG', 'RTX', 'T', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
    ])),
    2014: list(set([  # snapshot as of 2014-12-31, 30 tickers
        'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
        'PFE', 'PG', 'RTX', 'T', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
    ])),
    2015: list(set([  # snapshot as of 2015-12-31, 30 tickers
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
        'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
        'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
    ])),
    2016: list(set([  # snapshot as of 2016-12-31, 30 tickers
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
        'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
        'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
    ])),
    2017: list(set([  # snapshot as of 2017-12-31, 30 tickers
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
        'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
        'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'XOM',
    ])),
    2018: list(set([  # snapshot as of 2018-12-31, 30 tickers
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
        'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM',
    ])),
    2019: list(set([  # snapshot as of 2019-12-31, 30 tickers
        'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
        'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM',
    ])),
    2020: list(set([  # snapshot as of 2020-12-31, 30 tickers
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
    ])),
    2021: list(set([  # snapshot as of 2021-12-31, 30 tickers
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
    ])),
    2022: list(set([  # snapshot as of 2022-12-31, 30 tickers
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
    ])),
    2023: list(set([  # snapshot as of 2023-12-31, 30 tickers
        'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
        'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
        'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
    ])),
    2024: list(set([  # snapshot as of 2024-12-31, 30 tickers
        'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
        'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
        'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
    ])),
    2025: list(set([  # snapshot as of 2025-12-31, 30 tickers
        'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
        'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
        'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
    ])),
    2026: list(set([  # snapshot as of 2026-09-28, 30 tickers
        'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
        'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
        'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT',
    ])),
}


# ----------------------------------------------------------------------
# Data download - batched (not one ticker at a time) so hundreds of
# tickers take a handful of requests instead of hundreds of round-trips.
# ----------------------------------------------------------------------
def fetch_prices(tickers, start_date, batch_size=150):
    # SPEED FIX: this used to call yf.download() once PER TICKER,
    # sequentially, with a sleep between each - for 500-1000 tickers
    # that's one HTTP round-trip per name, easily 5-15 minutes on its
    # own. yfinance can fetch many tickers in ONE request (internally
    # threaded), so batching this into chunks cuts it down to a handful
    # of requests instead of hundreds. Chunked (not all-at-once) so one
    # bad/delisted ticker in the mix can't take down the whole download,
    # and so a very large universe doesn't hit a single oversized request.
    print(f"Downloading historical data for {len(tickers)} unique assets "
          f"across all yearly snapshots (batched, ~{batch_size}/request)...")
    query_map = {t: t.replace('.', '-') for t in tickers}
    reverse_map = {v: k for k, v in query_map.items()}
    all_series = []
    n_batches = -(-len(tickers) // batch_size)
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        query_tickers = [query_map[t] for t in batch]
        try:
            df = yf.download(query_tickers, start=start_date, auto_adjust=False,
                              progress=False, group_by='ticker', threads=True)
        except Exception as e:
            print(f"  Batch {i // batch_size + 1}/{n_batches} failed entirely, skipping: {e}")
            continue
        for query_ticker in query_tickers:
            ticker = reverse_map[query_ticker]
            try:
                sub = df[query_ticker] if isinstance(df.columns, pd.MultiIndex) else df
                if sub.empty:
                    continue
                col_to_use = 'Adj Close' if 'Adj Close' in sub.columns else 'Close'
                series = sub[col_to_use]
                if series.notna().any():
                    all_series.append(series.rename(ticker))
            except Exception:
                continue  # this one ticker had no usable data in the batch - skip it, not the batch
        print(f"  ...batch {i // batch_size + 1}/{n_batches} done "
              f"({min(i + batch_size, len(tickers))}/{len(tickers)} tickers)")
    data = pd.concat(all_series, axis=1).dropna(how='all') if all_series else pd.DataFrame()
    return data



# ----------------------------------------------------------------------
# Bad-data detection - see EXCLUDE_PRICE_ANOMALIES above.
# ----------------------------------------------------------------------
def detect_price_anomalies(data, max_monthly_ratio=MAX_MONTHLY_PRICE_RATIO):
    """Scans month-end prices for any ticker whose price multiplies (or
    divides) by more than `max_monthly_ratio` from one month to the next.
    Returns a DataFrame with one row per anomaly found: Ticker, Date,
    Prev Date, Prev Price, Price, Ratio - empty if none found."""
    monthly = data.resample('ME').last()
    anomalies = []
    for ticker in monthly.columns:
        series = monthly[ticker].dropna()
        if len(series) < 2:
            continue
        ratio = series / series.shift(1)
        bad_dates = ratio.index[(ratio > max_monthly_ratio) | (ratio < 1.0 / max_monthly_ratio)]
        for date in bad_dates:
            loc = series.index.get_loc(date)
            prev_date = series.index[loc - 1]
            anomalies.append({
                'Ticker': ticker,
                'Prev Date': prev_date.date(),
                'Prev Price': round(float(series.loc[prev_date]), 2),
                'Date': date.date(),
                'Price': round(float(series.loc[date]), 2),
                'Ratio': round(float(ratio.loc[date]), 2),
            })
    return pd.DataFrame(anomalies)


# ----------------------------------------------------------------------
# Reused-ticker protection - see RESTRICT_TO_MEMBERSHIP_WINDOW above.
# ----------------------------------------------------------------------
def restrict_to_membership_windows(data, universe_by_year, exempt=()):
    """Blanks every ticker's prices outside [Jan 1 of (first index year - 1),
    Dec 31 of (last index year + 1)], then drops columns left with no data.
    Returns (restricted_data, tickers_emptied)."""
    first_year, last_year = {}, {}
    for year, universe in universe_by_year.items():
        for t in universe:
            first_year[t] = min(first_year.get(t, year), year)
            last_year[t] = max(last_year.get(t, year), year)
    data = data.copy()
    for t in data.columns:
        if t in exempt or t not in first_year:
            continue
        outside = ((data.index < pd.Timestamp(f'{first_year[t] - 1}-01-01')) |
                   (data.index > pd.Timestamp(f'{last_year[t] + 1}-12-31')))
        data.loc[outside, t] = np.nan
    emptied = sorted(c for c in data.columns if data[c].isna().all())
    return data.drop(columns=emptied), emptied


# ----------------------------------------------------------------------
# Optional: drop the still-open current month for run-to-run reproducibility.
# ----------------------------------------------------------------------
def drop_incomplete_current_month(data, as_of=None):
    """Drops any rows of `data` that fall in the current REAL calendar
    month (as of `as_of`, default right now) - not the backtest's own
    notion of "current year". Returns (trimmed_data, n_rows_dropped)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    current_period = as_of.to_period('M')
    keep_mask = data.index.to_period('M') != current_period
    n_dropped = int((~keep_mask).sum())
    return data.loc[keep_mask], n_dropped


# ----------------------------------------------------------------------
# Sector lookup - purely informational (labels the momentum leaderboard
# in script 2), plays no part in any strategy's selection logic.
# ----------------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def get_sector_map(tickers):
    """Fetches (and caches to SECTOR_CACHE_FILE) each ticker's GICS
    sector from yfinance's .info. A ticker yfinance can't classify (or
    that errors out) falls back to 'Unknown' rather than crashing.

    BUG FIX: this used to fire hundreds of .info requests back-to-back
    with zero delay and no retry - Yahoo's rate limiter then starts
    rejecting requests with a 404 "quote not found" even for tickers
    that are definitely still trading (Bank of New York Mellon, Gap,
    Comerica, Walgreens, etc. all showed up as "missing" this way,
    which they very much aren't). A small delay between requests, plus
    one retry pass (after a longer pause) for anything that failed the
    first time, recovers almost all of those - true not-found tickers
    (actually delisted/renamed) still correctly fall back to 'Unknown'.
    This is purely cosmetic either way - sector never affects any
    strategy's selection or returns, only the leaderboard's labels."""
    cache = load_json(SECTOR_CACHE_FILE, {})
    missing = [t for t in tickers if t not in cache]
    if not missing:
        return cache

    print(f"Fetching sector classifications for {len(missing)} new ticker(s) "
          f"(cached from now on)...")
    failed_first_pass = []
    for t in missing:
        try:
            info = yf.Ticker(t).info
            sector = info.get('sector')
            if sector:
                cache[t] = sector
            else:
                failed_first_pass.append(t)
        except Exception:
            failed_first_pass.append(t)
        time.sleep(0.1)  # throttle - avoids tripping Yahoo's rate limiter

    if failed_first_pass:
        print(f"  {len(failed_first_pass)} ticker(s) failed on the first pass - many of these "
              f"are rate-limiting, not real 'not found' results - retrying once after a pause...")
        time.sleep(3)
        still_failed = []
        for t in failed_first_pass:
            try:
                info = yf.Ticker(t).info
                sector = info.get('sector')
                cache[t] = sector or 'Unknown'
                if not sector:
                    still_failed.append(t)
            except Exception:
                cache[t] = 'Unknown'
                still_failed.append(t)
            time.sleep(0.2)
        if still_failed:
            print(f"  {len(still_failed)} ticker(s) still unclassified after retry - marked 'Unknown' "
                  f"(cosmetic only): {', '.join(still_failed[:20])}{'...' if len(still_failed) > 20 else ''}")
        recovered = len(failed_first_pass) - len(still_failed)
        if recovered:
            print(f"  Recovered {recovered} ticker(s) on retry - confirms those were rate-limiting, "
                  f"not genuinely missing.")

    save_json(SECTOR_CACHE_FILE, cache)
    return cache

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    active_dow_by_year = {y: u for y, u in dow_by_year.items() if y in RUN_YEARS}
    if not active_dow_by_year:
        raise SystemExit(f"RUN_YEARS={RUN_YEARS} matches none of the years in dow_by_year "
                          f"({sorted(dow_by_year.keys())}). Fix RUN_YEARS and rerun.")
    data_start_date = f'{min(active_dow_by_year) - 1}-01-01'
    print(f"RUN_YEARS={RUN_YEARS} - downloading data from {data_start_date} onward.")

    all_tickers = sorted(set(
        [t for universe in active_dow_by_year.values() for t in universe] + [BENCHMARK_TICKER]))
    data = fetch_prices(all_tickers, data_start_date)
    if data.empty:
        raise SystemExit("No price data returned - check network access / tickers. "
                          "(This script needs real yfinance access - it will NOT work "
                          "in a network-sandboxed environment.)")

    reused = [t for t in KNOWN_REUSED_TICKERS if t in data.columns]
    if reused:
        print(f"\nDropping {len(reused)} known reused ticker(s) - yfinance returns a "
              f"different company's prices for them:")
        for t in reused:
            print(f"  {t}: {KNOWN_REUSED_TICKERS[t]}")
        data = data.drop(columns=reused)

    if RESTRICT_TO_MEMBERSHIP_WINDOW:
        data, emptied = restrict_to_membership_windows(
            data, active_dow_by_year, exempt={BENCHMARK_TICKER})
        if emptied:
            print(f"\n{len(emptied)} ticker(s) had data ONLY outside their index membership "
                  f"window (+/- 1 year) - that data belongs to whoever reused the ticker "
                  f"later, so they're treated as missing (RESTRICT_TO_MEMBERSHIP_WINDOW=True): "
                  f"{', '.join(emptied)}")

    anomalies_df = detect_price_anomalies(data)
    if not anomalies_df.empty:
        bad_tickers = sorted(anomalies_df['Ticker'].unique())
        print(f"\n\u26a0\ufe0f  WARNING: found {len(anomalies_df)} implausible month-over-month price "
              f"jump(s) (more than {MAX_MONTHLY_PRICE_RATIO:.0f}x, or less than "
              f"1/{MAX_MONTHLY_PRICE_RATIO:.0f}x, in a single month) across "
              f"{len(bad_tickers)} ticker(s) - almost certainly bad source data from "
              f"yfinance, not a real market move:")
        print(anomalies_df.to_string(index=False))
        if EXCLUDE_PRICE_ANOMALIES:
            print(f"Dropping {', '.join(bad_tickers)} entirely from this run "
                  f"(EXCLUDE_PRICE_ANOMALIES=True) so they can't distort momentum "
                  f"rankings or any downstream calculation.")
            data = data.drop(columns=[t for t in bad_tickers if t in data.columns])

    if EXCLUDE_INCOMPLETE_CURRENT_MONTH:
        data, n_dropped = drop_incomplete_current_month(data)
        if n_dropped:
            print(f"Dropped {n_dropped} row(s) from the still-open current month "
                  f"(EXCLUDE_INCOMPLETE_CURRENT_MONTH=True).")

    sectors = get_sector_map(all_tickers)

    missing_tickers = sorted(set(all_tickers) - set(data.columns))
    fetch_timestamp = pd.Timestamp(now_eastern())  # Eastern time (handles EST/EDT)
    print(f"\n{'='*100}\n  DATA FINGERPRINT - this is what gets cached; the other three scripts will "
          f"print this same block when they load it\n{'='*100}")
    print(f"Fetched at: {fetch_timestamp}")
    print(f"Requested {len(all_tickers)} tickers, got usable data for {len(data.columns)} "
          f"({len(missing_tickers)} missing/empty).")
    if missing_tickers:
        print(f"  Missing: {', '.join(missing_tickers[:30])}"
              f"{' ...' if len(missing_tickers) > 30 else ''}")
    print(f"Date range fetched: {data.index.min().date()} to {data.index.max().date()} "
          f"({len(data.index)} trading days)")
    if BENCHMARK_TICKER in data.columns:
        bench = data[BENCHMARK_TICKER].dropna()
        checkpoints = [bench.index.min()]
        for y in range(bench.index.min().year, bench.index.max().year + 1, 3):
            candidates = bench.index[bench.index.year == y]
            if len(candidates) > 0:
                checkpoints.append(candidates[-1])
        checkpoints.append(bench.index.max())
        print(f"{BENCHMARK_TICKER} reference closes (should be IDENTICAL every time you load this cache):")
        for dt in sorted(set(checkpoints)):
            print(f"  {dt.date()}: ${bench.loc[dt]:,.4f}")
    print(f"{'='*100}\n")

    cache = {
        'data': data,
        'sp500_by_year': active_dow_by_year,  # key name kept for scripts 2-4
        'sectors': sectors,
        'run_years': RUN_YEARS,
        'benchmark_ticker': BENCHMARK_TICKER,
        'fetch_timestamp': fetch_timestamp,
        'missing_tickers': missing_tickers,
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    size_kb = os.path.getsize(CACHE_FILE) / 1024
    print(f"Saved cache: {CACHE_FILE} ({size_kb:,.1f} KB)")
    print(f"\nNow run 2_topn_compare.py, 3_buffer_strategies.py, and/or 4_live_recommendation.py - "
          f"they'll load this cache instantly instead of re-downloading.")


if __name__ == '__main__':
    main()
