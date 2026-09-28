# -*- coding: utf-8 -*-
"""
1b_fetch_and_cache_ndx100.py
============================
NASDAQ-100 VERSION of step 1 (the S&P 500 version is 1_fetch_and_cache_data.py).
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
the ndx100_by_year universe (filtered to RUN_YEARS, stored under the
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
RUN_YEARS = list(range(2007, 2027))  # ONE LINE controls the whole pipeline's
                                    # span - all three downstream scripts
                                    # inherit this through the cache file.
                                    # e.g. list(range(2015, 2027)) for 2015
                                    # onward, or list(ndx100_by_year.keys())
                                    # for the full 2007-2026 history.
BENCHMARK_TICKER = 'QQQ'
DATA_START_DATE_FALLBACK = '2006-01-01'  # only used if RUN_YEARS is somehow
                                    # empty; normally overridden below to
                                    # f'{min(RUN_YEARS)-1}-01-01' so a
                                    # narrow RUN_YEARS downloads a lot less.
SECTOR_CACHE_FILE = 'ndx100_sector_cache.json'  # caches each ticker's GICS
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
    'BBBY': 'Bed Bath & Beyond (bankrupt 2023) - ticker taken by Beyond Inc '
            '(ex-Overstock) Aug 2025, so yfinance returns Overstock prices back to 2002',
    'GOLD': 'Randgold Resources (Nasdaq-100 2011-12, acquired by Barrick 2019) - '
            'GOLD was Barrick\'s NYSE ticker until May 2025, then vacated',
}
CACHE_FILE_NAME = 'momentum_data_cache_ndx100.pkl'  # deliberately DIFFERENT from the
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
# The NASDAQ-100 index membership, one snapshot per calendar year (point
# in time): the last composition on or before each year-end, taken from
# floyds1995/Auto-Index-Constituents-Tracker on GitHub (built from
# Wikipedia's Nasdaq-100 change log, which begins Feb 2007 - so there is
# no earlier history from this source). NOT an official record.
#
# Old tickers are replaced by the ticker yfinance knows the same company
# by today (checked against company press releases / SEC filings):
#   UAUA->UAL  RIMM->BB (2013)  FI->FISV (Fiserv, 2025)  HANS->MNST (2012)
#   MICC->TIGO (2019)  VIP->VEON (2017)  NLOK->GEN  MYL->VTRS (2020)
#   DISCA/DISCK->WBD (2022)  VIAB->PSKY (via VIAC/PARA)
# The source already labels some companies with a later ticker (BKNG, META,
# WTW...), so those needed no change. Not mapped, so they will show as
# missing or partial: FOXA/FOX (Fox Corp only dates from 2013/2019), IACI,
# LMCA/LMCK (Liberty tracking stocks), SIRI (post-2024 merger history).
# ----------------------------------------------------------------------
ndx100_by_year = {
    2007: list(set([  # snapshot as of 2007-12-24, 101 tickers
        'AAPL', 'ADBE', 'ADSK', 'AKAM', 'ALTR', 'AMAT', 'AMGN', 'AMLN', 'AMZN', 'APOL',
        'ATVI', 'BBBY', 'BEAS', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CELG', 'CEPH', 'CHKP',
        'CHRW', 'COST', 'CSCO', 'CTAS', 'CTSH', 'CTXS', 'DELL', 'WBD', 'DISH', 'EA',
        'EBAY', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FISV', 'FLEX', 'FMCN', 'FOXA', 'FWLT',
        'GENZ', 'GILD', 'GOOGL', 'GRMN', 'MNST', 'HOLX', 'HSIC', 'IACI', 'INFY', 'INTC',
        'INTU', 'ISRG', 'JAVA', 'JNPR', 'JOYG', 'KLAC', 'LAMR', 'LBTYA', 'LEAP', 'LLTC',
        'LMCK', 'LOGI', 'LRCX', 'LVLT', 'MCHP', 'MDLZ', 'TIGO', 'MRVL', 'MSFT', 'NIHD',
        'GEN', 'NTAP', 'NVDA', 'ORCL', 'PAYX', 'PCAR', 'PDCO', 'PETM', 'QCOM', 'QRTEA',
        'BB', 'RYAAY', 'SBUX', 'SHLD', 'SIAL', 'SIRI', 'SNDK', 'SPLS', 'SRCL', 'STLD',
        'TEVA', 'TLAB', 'UAL', 'VMED', 'VRSN', 'VRTX', 'WFMI', 'WYNN', 'XLNX', 'XRAY',
        'YHOO',
    ])),
    2008: list(set([  # snapshot as of 2008-12-22, 103 tickers
        'AAPL', 'ADBE', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'AMAT', 'AMGN', 'AMZN', 'APOL',
        'ATVI', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'CA', 'CELG', 'CEPH', 'CHKP', 'CHRW',
        'COST', 'CSCO', 'CTAS', 'CTSH', 'CTXS', 'DELL', 'WBD', 'DISH', 'DTV', 'EA',
        'EBAY', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FISV', 'FLEX', 'FLIR', 'FMCN', 'FOXA',
        'FSLR', 'FWLT', 'GENZ', 'GILD', 'GOOGL', 'GRMN', 'MNST', 'HOLX', 'HSIC', 'IACI',
        'ILMN', 'INFY', 'INTC', 'INTU', 'ISRG', 'JAVA', 'JBHT', 'JNPR', 'JOYG', 'KLAC',
        'LBTYA', 'LIFE', 'LLTC', 'LMCK', 'LOGI', 'LRCX', 'MCHP', 'MDLZ', 'TIGO', 'MRVL',
        'MSFT', 'MXIM', 'NIHD', 'GEN', 'NTAP', 'NVDA', 'ORCL', 'ORLY', 'PAYX', 'PCAR',
        'PDCO', 'PPDI', 'QCOM', 'QRTEA', 'BB', 'ROST', 'RYAAY', 'SBUX', 'SHLD', 'SIAL',
        'SPLS', 'SRCL', 'STLD', 'STX', 'TEVA', 'URBN', 'VRSN', 'VRTX', 'WCRX', 'WYNN',
        'XLNX', 'XRAY', 'YHOO',
    ])),
    2009: list(set([  # snapshot as of 2009-12-21, 101 tickers
        'AAPL', 'ADBE', 'ADP', 'ADSK', 'ALTR', 'AMAT', 'AMGN', 'AMZN', 'APOL', 'ATVI',
        'BBBY', 'BIDU', 'BIIB', 'BKNG', 'BMC', 'CA', 'CELG', 'CEPH', 'CERN', 'CHKP',
        'CHRW', 'COST', 'CSCO', 'CTAS', 'CTSH', 'CTXS', 'DELL', 'WBD', 'DISH', 'DTV',
        'EA', 'EBAY', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FISV', 'FLEX', 'FLIR', 'FOXA',
        'FSLR', 'FWLT', 'GENZ', 'GILD', 'GOOGL', 'GRMN', 'HOLX', 'HSIC', 'ILMN', 'INFY',
        'INTC', 'INTU', 'ISRG', 'JBHT', 'JOYG', 'KLAC', 'LIFE', 'LLTC', 'LMCK', 'LOGI',
        'LRCX', 'MAT', 'MCHP', 'MDLZ', 'TIGO', 'MRVL', 'MSFT', 'MXIM', 'VTRS', 'NIHD',
        'GEN', 'NTAP', 'NVDA', 'ORCL', 'ORLY', 'PAYX', 'PCAR', 'PDCO', 'QCOM', 'QGEN',
        'QRTEA', 'BB', 'ROST', 'SBUX', 'SHLD', 'SIAL', 'SNDK', 'SPLS', 'SRCL', 'STX',
        'TEVA', 'URBN', 'VMED', 'VOD', 'VRSN', 'VRTX', 'WCRX', 'WYNN', 'XLNX', 'XRAY',
        'YHOO',
    ])),
    2010: list(set([  # snapshot as of 2010-12-20, 101 tickers
        'AAPL', 'ADBE', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'AMAT', 'AMGN', 'AMZN', 'APOL',
        'ATVI', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'BMC', 'CA', 'CELG', 'CEPH', 'CERN',
        'CHKP', 'CHRW', 'COST', 'CSCO', 'CTSH', 'CTXS', 'DELL', 'WBD', 'DLTR', 'DTV',
        'EA', 'EBAY', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FFIV', 'FISV', 'FLEX', 'FLIR',
        'FOXA', 'FSLR', 'GENZ', 'GILD', 'GOOGL', 'GRMN', 'HSIC', 'ILMN', 'INFY', 'INTC',
        'INTU', 'ISRG', 'JOYG', 'KLAC', 'LIFE', 'LLTC', 'LMCK', 'LRCX', 'MAT', 'MCHP',
        'MDLZ', 'TIGO', 'MRVL', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NFLX', 'NIHD', 'GEN',
        'NTAP', 'NVDA', 'ORCL', 'ORLY', 'PAYX', 'PCAR', 'QCOM', 'QGEN', 'QRTEA', 'BB',
        'ROST', 'SBUX', 'SHLD', 'SIAL', 'SNDK', 'SPLS', 'SRCL', 'STX', 'TCOM', 'TEVA',
        'URBN', 'VMED', 'VOD', 'VRSN', 'VRTX', 'WCRX', 'WFM', 'WYNN', 'XLNX', 'XRAY',
        'YHOO',
    ])),
    2011: list(set([  # snapshot as of 2011-12-19, 101 tickers
        'AAPL', 'ADBE', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'ALXN', 'AMAT', 'AMGN', 'AMZN',
        'APOL', 'ATVI', 'AVGO', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'BMC', 'CA', 'CELG',
        'CERN', 'CHKP', 'CHRW', 'COST', 'CSCO', 'CTSH', 'CTXS', 'DELL', 'WBD', 'DLTR',
        'DTV', 'EA', 'EBAY', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FFIV', 'FISV', 'FLEX',
        'FOSL', 'FOXA', 'FSLR', 'GILD', 'GMCR', 'GOLD', 'GOOGL', 'GRMN', 'HSIC', 'INFY',
        'INTC', 'INTU', 'ISRG', 'KLAC', 'LIFE', 'LLTC', 'LMCK', 'LRCX', 'MAT', 'MCHP',
        'MDLZ', 'MNST', 'MRVL', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NFLX', 'GEN', 'NTAP',
        'NUAN', 'NVDA', 'ORCL', 'ORLY', 'PAYX', 'PCAR', 'PRGO', 'QCOM', 'QRTEA', 'BB',
        'ROST', 'SBUX', 'SHLD', 'SIAL', 'SIRI', 'SNDK', 'SPLS', 'SRCL', 'STX', 'TCOM',
        'TEVA', 'VMED', 'VOD', 'VRSN', 'VRTX', 'WCRX', 'WFM', 'WYNN', 'XLNX', 'XRAY',
        'YHOO',
    ])),
    2012: list(set([  # snapshot as of 2012-12-24, 99 tickers
        'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'ALXN', 'AMAT', 'AMGN',
        'AMZN', 'ATVI', 'AVGO', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'BMC', 'CA', 'CELG',
        'CERN', 'CHKP', 'CHRW', 'COST', 'CSCO', 'CTRX', 'CTSH', 'CTXS', 'DELL', 'WBD',
        'DLTR', 'DTV', 'EBAY', 'EQIX', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FFIV', 'FISV',
        'FOSL', 'FOXA', 'GILD', 'GOLD', 'GOOGL', 'GRMN', 'HSIC', 'INTC', 'INTU', 'ISRG',
        'KLAC', 'LBTYA', 'LIFE', 'LLTC', 'LMCA', 'LMCK', 'MAT', 'MCHP', 'MDLZ', 'META',
        'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS', 'GEN', 'NTAP', 'NUAN', 'NVDA', 'ORCL',
        'ORLY', 'PAYX', 'PCAR', 'PRGO', 'QCOM', 'QRTEA', 'REGN', 'ROST', 'SBAC', 'SBUX',
        'SHLD', 'SIAL', 'SIRI', 'SNDK', 'SPLS', 'SRCL', 'STX', 'TXN', 'PSKY', 'VMED',
        'VOD', 'VRSK', 'VRTX', 'WDC', 'WFM', 'WYNN', 'XLNX', 'XRAY', 'YHOO',
    ])),
    2013: list(set([  # snapshot as of 2013-12-23, 99 tickers
        'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'ALXN', 'AMAT', 'AMGN',
        'AMZN', 'ATVI', 'AVGO', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'CA', 'CELG', 'CERN',
        'CHKP', 'CHRW', 'CHTR', 'COST', 'CSCO', 'CTRX', 'CTSH', 'CTXS', 'WBD', 'DISH',
        'DLTR', 'DTV', 'EBAY', 'EQIX', 'ESRX', 'EXPD', 'EXPE', 'FAST', 'FFIV', 'FISV',
        'FOXA', 'GILD', 'GMCR', 'GOOGL', 'GRMN', 'HSIC', 'ILMN', 'INTC', 'INTU', 'ISRG',
        'KLAC', 'KRFT', 'LBTYA', 'LLTC', 'LMCA', 'LMCK', 'MAR', 'MAT', 'MDLZ', 'META',
        'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NFLX', 'GEN', 'NTAP', 'NVDA', 'NXPI',
        'ORLY', 'PAYX', 'PCAR', 'QCOM', 'QRTEA', 'REGN', 'ROST', 'SBAC', 'SBUX', 'SIAL',
        'SIRI', 'SNDK', 'SPLS', 'SRCL', 'STX', 'TRIP', 'TSCO', 'TSLA', 'TXN', 'PSKY',
        'VEON', 'VOD', 'VRSK', 'VRTX', 'WDC', 'WFM', 'WYNN', 'XLNX', 'YHOO',
    ])),
    2014: list(set([  # snapshot as of 2014-12-22, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALTR', 'ALXN', 'AMAT',
        'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'CA', 'CELG',
        'CERN', 'CHKP', 'CHRW', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CTRX', 'CTSH', 'CTXS',
        'WBD', 'DISH', 'DLTR', 'DTV', 'EA', 'EBAY', 'EQIX', 'ESRX', 'EXPD', 'FAST',
        'FISV', 'FOX', 'FOXA', 'GILD', 'GMCR', 'GOOG', 'GOOGL', 'GRMN', 'HSIC', 'ILMN',
        'INTC', 'INTU', 'ISRG', 'KLAC', 'KRFT', 'LBTYA', 'LBTYK', 'LLTC', 'LMCA', 'LMCK',
        'LRCX', 'MAR', 'MAT', 'MDLZ', 'META', 'MNST', 'MSFT', 'MU', 'VTRS', 'NFLX',
        'GEN', 'NTAP', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'QCOM', 'QRTEA', 'REGN',
        'ROST', 'SBAC', 'SBUX', 'SIAL', 'SIRI', 'SNDK', 'SPLS', 'SRCL', 'STX', 'TRIP',
        'TSCO', 'TSLA', 'TXN', 'PSKY', 'VEON', 'VOD', 'VRSK', 'VRTX', 'WDC', 'WFM',
        'WYNN', 'XLNX', 'YHOO',
    ])),
    2015: list(set([  # snapshot as of 2015-12-21, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALXN', 'AMAT', 'AMGN',
        'AMZN', 'ATVI', 'BBBY', 'BIDU', 'BIIB', 'BKNG', 'BMRN', 'CA', 'CELG', 'CERN',
        'CHKP', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CTSH', 'CTXS', 'WBD', 'DISH', 'DLTR',
        'EA', 'EBAY', 'ENDP', 'ESRX', 'EXPE', 'FAST', 'FISV', 'FOX', 'FOXA', 'GILD',
        'GOOG', 'GOOGL', 'HSIC', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'JD', 'KHC',
        'KLAC', 'LBTYA', 'LBTYK', 'LLTC', 'LMCA', 'LMCK', 'LRCX', 'MAR', 'MAT', 'MDLZ',
        'META', 'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NCLH', 'NFLX', 'GEN', 'NTAP',
        'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'PYPL', 'QCOM', 'QRTEA', 'REGN', 'ROST',
        'SBAC', 'SBUX', 'SIRI', 'SNDK', 'SRCL', 'STX', 'SWKS', 'TCOM', 'TMUS', 'TRIP',
        'TSCO', 'TSLA', 'TXN', 'ULTA', 'PSKY', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WDC',
        'WFM', 'XLNX', 'YHOO',
    ])),
    2016: list(set([  # snapshot as of 2016-12-19, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALXN', 'AMAT', 'AMGN',
        'AMZN', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'BMRN', 'CA', 'CELG', 'CERN',
        'CHKP', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'WBD',
        'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE', 'FAST', 'FISV', 'FOX', 'FOXA',
        'GILD', 'GOOG', 'GOOGL', 'HAS', 'HOLX', 'HSIC', 'ILMN', 'INCY', 'INTC', 'INTU',
        'ISRG', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LBTYK', 'LRCX', 'MAR', 'MAT', 'MCHP',
        'MDLZ', 'META', 'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NCLH', 'NFLX', 'GEN',
        'NTES', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'PYPL', 'QCOM', 'QRTEA', 'REGN',
        'ROST', 'SBAC', 'SBUX', 'SHPG', 'SIRI', 'STX', 'SWKS', 'TCOM', 'TMUS', 'TRIP',
        'TSCO', 'TSLA', 'TXN', 'ULTA', 'PSKY', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WDC',
        'XLNX', 'XRAY', 'YHOO',
    ])),
    2017: list(set([  # snapshot as of 2017-12-18, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMGN',
        'AMZN', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'BMRN', 'CA', 'CDNS',
        'CELG', 'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH',
        'CTXS', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE', 'FAST', 'FISV', 'FOX',
        'FOXA', 'GILD', 'GOOG', 'GOOGL', 'HAS', 'HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY',
        'INTC', 'INTU', 'ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LBTYK', 'LRCX',
        'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS',
        'NFLX', 'GEN', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PYPL', 'QCOM', 'QRTEA',
        'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'SNPS', 'STX', 'SWKS', 'TCOM', 'TMUS',
        'TSLA', 'TTWO', 'TXN', 'ULTA', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WDAY', 'WDC',
        'WYNN', 'XLNX', 'XRAY',
    ])),
    2018: list(set([  # snapshot as of 2018-12-24, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'BMRN', 'CDNS',
        'CELG', 'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH',
        'CTXS', 'DLTR', 'EA', 'EBAY', 'EXPE', 'FAST', 'FISV', 'FOX', 'FOXA', 'GILD',
        'GOOG', 'GOOGL', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
        'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LBTYK', 'LRCX', 'LULU', 'MAR', 'MCHP',
        'MDLZ', 'MELI', 'META', 'MNST', 'MSFT', 'MU', 'MXIM', 'VTRS', 'NFLX', 'GEN',
        'NTAP', 'NTES', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'PYPL', 'QCOM',
        'REGN', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TCOM', 'TMUS', 'TSLA', 'TTWO',
        'TXN', 'UAL', 'ULTA', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'WDC', 'WTW',
        'WYNN', 'XEL', 'XLNX',
    ])),
    2019: list(set([  # snapshot as of 2019-12-23, 103 tickers
        'AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'BMRN',
        'CDNS', 'CDW', 'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CSCO', 'CSGP',
        'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'EXC', 'EXPE', 'FAST',
        'FISV', 'FOX', 'FOXA', 'GILD', 'GOOG', 'GOOGL', 'IDXX', 'ILMN', 'INCY', 'INTC',
        'INTU', 'ISRG', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LBTYK', 'LRCX', 'LULU', 'MAR',
        'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MSFT', 'MU', 'MXIM', 'NFLX', 'NTAP',
        'NTES', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'PYPL', 'QCOM', 'REGN',
        'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'SPLK', 'SWKS', 'TCOM', 'TMUS', 'TSLA',
        'TTWO', 'TXN', 'UAL', 'ULTA', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'WDC',
        'WTW', 'XEL', 'XLNX',
    ])),
    2020: list(set([  # snapshot as of 2020-12-21, 102 tickers
        'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'ALXN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'CDNS',
        'CDW', 'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CSCO', 'CSX', 'CTAS',
        'CTSH', 'DLTR', 'DOCU', 'DXCM', 'EA', 'EBAY', 'EXC', 'FAST', 'FISV', 'FOX',
        'FOXA', 'GILD', 'GOOG', 'GOOGL', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
        'JD', 'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI',
        'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MTCH', 'MU', 'MXIM', 'NFLX', 'NTES',
        'NVDA', 'NXPI', 'OKTA', 'ORLY', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PTON', 'PYPL',
        'QCOM', 'REGN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'SPLK', 'SWKS', 'TCOM',
        'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'XEL',
        'XLNX', 'ZM',
    ])),
    2021: list(set([  # snapshot as of 2021-12-20, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'CDNS',
        'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DDOG',
        'DLTR', 'DOCU', 'DXCM', 'EA', 'EBAY', 'EXC', 'FAST', 'FISV', 'FTNT', 'GILD',
        'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP',
        'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META',
        'MNST', 'MRNA', 'MRVL', 'MSFT', 'MTCH', 'MU', 'NFLX', 'NTES', 'NVDA', 'NXPI',
        'OKTA', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PTON', 'PYPL', 'QCOM',
        'REGN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'SPLK', 'SWKS', 'TEAM', 'TMUS',
        'TSLA', 'TXN', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'XEL', 'XLNX', 'ZM',
        'ZS',
    ])),
    2022: list(set([  # snapshot as of 2022-12-19, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR',
        'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX',
        'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EBAY', 'ENPH', 'EXC', 'FANG',
        'FAST', 'FISV', 'FTNT', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN',
        'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU',
        'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU',
        'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP',
        'PYPL', 'QCOM', 'REGN', 'RIVN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'TEAM',
        'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZM',
        'ZS',
    ])),
    2023: list(set([  # snapshot as of 2023-12-18, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
        'AMZN', 'ANSS', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 'CDNS',
        'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX',
        'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST',
        'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC',
        'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDB',
        'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NVDA',
        'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL',
        'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'SPLK', 'TEAM', 'TMUS',
        'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL',
        'ZS',
    ])),
    2024: list(set([  # snapshot as of 2024-12-23, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
        'AMZN', 'ANSS', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB', 'BKNG',
        'BKR', 'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD',
        'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DXCM', 'EA', 'EXC',
        'FANG', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX',
        'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR',
        'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRVL', 'MSFT', 'MSTR', 'MU',
        'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
        'PEP', 'PLTR', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SNPS', 'TEAM',
        'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL',
        'ZS',
    ])),
    2025: list(set([  # snapshot as of 2025-12-22, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALNY', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'AZN', 'BKNG', 'BKR',
        'CCEP', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP',
        'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST',
        'FER', 'FTNT', 'GEHC', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'INSM', 'INTC',
        'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'MAR', 'MCHP', 'MDLZ',
        'MELI', 'META', 'MNST', 'MPWR', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NFLX', 'NVDA',
        'NXPI', 'ODFL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PLTR', 'PYPL',
        'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SHOP', 'SNPS', 'STX', 'TEAM', 'TMUS',
        'TRI', 'TSLA', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'WDC', 'XEL',
        'ZS',
    ])),
    2026: list(set([  # snapshot as of 2026-09-14, 101 tickers
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALAB', 'ALNY', 'AMAT',
        'AMD', 'AMGN', 'AMZN', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'BKNG', 'BKR',
        'CCEP', 'CDNS', 'CEG', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CRWV', 'CSCO', 'CSX',
        'CTAS', 'DASH', 'DDOG', 'DXCM', 'EXC', 'FANG', 'FAST', 'FER', 'FTNT', 'GEHC',
        'GILD', 'GOOG', 'GOOGL', 'HON', 'HONA', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP',
        'KLAC', 'LIN', 'LITE', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST',
        'MPWR', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NBIS', 'NFLX', 'NVDA', 'NXPI', 'ODFL',
        'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PLTR', 'PYPL', 'QCOM', 'REGN',
        'RKLB', 'ROP', 'ROST', 'SBUX', 'SHOP', 'SNDK', 'SNPS', 'SPCX', 'STX', 'TER',
        'TMUS', 'TRI', 'TSLA', 'TTWO', 'TXN', 'VRTX', 'WBD', 'WDAY', 'WDC', 'WMT',
        'XEL',
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
    active_ndx100_by_year = {y: u for y, u in ndx100_by_year.items() if y in RUN_YEARS}
    if not active_ndx100_by_year:
        raise SystemExit(f"RUN_YEARS={RUN_YEARS} matches none of the years in ndx100_by_year "
                          f"({sorted(ndx100_by_year.keys())}). Fix RUN_YEARS and rerun.")
    data_start_date = f'{min(active_ndx100_by_year) - 1}-01-01'
    print(f"RUN_YEARS={RUN_YEARS} - downloading data from {data_start_date} onward.")

    all_tickers = sorted(set(
        [t for universe in active_ndx100_by_year.values() for t in universe] + [BENCHMARK_TICKER]))
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
            data, active_ndx100_by_year, exempt={BENCHMARK_TICKER})
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
        'sp500_by_year': active_ndx100_by_year,  # key name kept for scripts 2-4
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
