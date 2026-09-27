# -*- coding: utf-8 -*-
"""
1_fetch_and_cache_data.py
===========================
STEP 1 OF 4. Run this FIRST, whenever you want fresh data (once a day is
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
the sp500_by_year universe (filtered to RUN_YEARS), the sector lookup,
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
RUN_YEARS = list(range(2008, 2027))  # ONE LINE controls the whole pipeline's
                                    # span - all three downstream scripts
                                    # inherit this through the cache file.
                                    # e.g. list(range(2015, 2027)) for 2015
                                    # onward, or list(sp500_by_year.keys())
                                    # for the full 2008-2026 history.
BENCHMARK_TICKER = 'SPY'
DATA_START_DATE_FALLBACK = '2007-01-01'  # only used if RUN_YEARS is somehow
                                    # empty; normally overridden below to
                                    # f'{min(RUN_YEARS)-1}-01-01' so a
                                    # narrow RUN_YEARS downloads a lot less.
SECTOR_CACHE_FILE = 'sp500_sector_cache.json'  # caches each ticker's GICS
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
CACHE_FILE_NAME = 'momentum_data_cache.pkl'  # MUST match the CACHE_FILE_NAME
                                    # the other three scripts look for -
                                    # don't rename this independently.

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
# The S&P 500 index membership, one snapshot per calendar year (point in
# time - see the ORIGINAL notebook this section was carried over from for
# the full explanation of where this data comes from and its caveats:
# Wikipedia's community-maintained "S&P 500 component changes" page,
# reconstructed year by year - NOT an official, guaranteed-complete
# record).
#
# Hidden renames: the same company kept trading under a new ticker, and
# yfinance only knows the new one, so the old ticker came back empty and
# the company was split into two half-histories. Each old ticker below is
# replaced by the new one in every year (checked against company press
# releases / SEC filings):
#   FB->META (2022)  BBT->TFC (2019)  HRS->LHX (2019)  TMK->GL (2019)
#   JEC->J (2019)  UTX->RTX (2020)  DWDP->DD (2019)  BHGE->BKR (2019)
#   SYMC->NLOK->GEN (2019/2022)  CTL->LUMN (2020)  COG->CTRA (2021)
#   LB->BBWI (2021)  FBHS->FBIN (2022)  ADS->BFH (2022)  GPS->GAP (2024)
#   DISCA->WBD (2022)  PKI->RVTY (2023)  RE->EG (2023)  CDAY->DAY (2024)
#   FLT->CPAY (2024)  HCP->PEAK->DOC (2019/2024)  FI->FISV (2025)
#   CBS->VIAC->PARA->PSKY (2019/2022/2025)  WYND->TNL (2021)
#   DDR->SITC (2018)  FII->FHI (2020)  HFC->DINO (2022)
#   Merger successors (yfinance may or may not carry the pre-merger
#   history, but the old ticker returns nothing either way):
#   MYL->VTRS (2020)  PX->LIN (2018)
# ----------------------------------------------------------------------
sp500_by_year = {
    2008: list(set([  # snapshot as of 2008-12-29, 498 tickers
        'A', 'AABA', 'AAPL', 'COR', 'ABT', 'ACAS', 'ACS', 'ADBE', 'ADI', 'ADM',
        'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV',
        'AIZ', 'AKAM', 'AKS', 'ALL', 'ALTR', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT',
        'AMZN', 'AN', 'ANDV', 'ANF', 'ELV', 'AON', 'APA', 'APC', 'APD', 'APH',
        'APOL', 'HWM', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AYE', 'AZO', 'BA',
        'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDK', 'BDX', 'BEAM', 'BEN',
        'BF.B', 'BKR', 'BIG', 'BIIB', 'BJS', 'BNY', 'BALL', 'BMC', 'BMS', 'BMY',
        'BNI', 'BRCM', 'BSX', 'BTUUQ', 'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM',
        'CAT', 'CB', 'CBE', 'CBRE', 'PSKY', 'CCE', 'CCL', 'CEG', 'CELG', 'CEPH',
        'CF', 'CHK', 'CHRW', 'CI', 'CIEN', 'CINF', 'CITGQ', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL', 'COP',
        'COST', 'COV', 'CPB', 'CPWR', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH',
        'CTX', 'CTXS', 'CVG', 'CVH', 'CVS', 'CVX', 'D', 'DD', 'SITC', 'DE',
        'DELL', 'DF', 'DFS', 'DGX', 'DHI', 'DHR', 'DIS', 'DNB', 'DOV', 'DOW',
        'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'DXC', 'DYN', 'EA', 'EBAY',
        'ECL', 'ED', 'EFX', 'EIX', 'EKDKQ', 'EL', 'EMC', 'EMN', 'EMR', 'EOG',
        'EP', 'EQ', 'EQR', 'EQT', 'ESRX', 'ESV', 'ETFC', 'ETN', 'ETR', 'EXC',
        'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE', 'FHN', 'FHI',
        'FIS', 'FISV', 'FITB', 'FLR', 'FLS', 'FOXA', 'FRX', 'FTR', 'GAS', 'GD',
        'GE', 'GENZ', 'GHC', 'GILD', 'GIS', 'GLW', 'GME', 'GNW', 'GOOGL', 'GPC',
        'GAP', 'GR', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK',
        'DOC', 'HD', 'HES', 'HIG', 'HNZ', 'HOG', 'HON', 'HOT', 'HPQ', 'HRB',
        'LHX', 'HSH', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT',
        'INTC', 'INTU', 'IP', 'IPG', 'IR', 'ISRG', 'ITT', 'ITW', 'IVZ', 'JAVA',
        'JBL', 'JCI', 'JCP', 'J', 'JEF', 'JNJ', 'JNPR', 'JNS', 'JNY', 'JPM',
        'JWN', 'K', 'KBH', 'KDP', 'KEY', 'KG', 'KIM', 'KLAC', 'KMB', 'KO',
        'KR', 'KSS', 'L', 'BBWI', 'LEG', 'LEN', 'LH', 'LIFE', 'LLL', 'LLTC',
        'LLY', 'LM', 'LMT', 'LNC', 'LO', 'LOW', 'LSI', 'LUV', 'LXK', 'M',
        'MA', 'MAR', 'MAS', 'MAT', 'MBI', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ',
        'MDP', 'MDT', 'MEE', 'MER', 'MET', 'MFE', 'MHS', 'MI', 'MIL', 'MKC',
        'MRSH', 'MMM', 'MO', 'MOLX', 'MON', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI',
        'MTB', 'MTLQQ', 'MTW', 'MU', 'MUR', 'MWV', 'MWW', 'VTRS', 'NBL', 'NBR',
        'NCC', 'NDAQ', 'NE', 'NEE', 'NEM', 'NI', 'NKE', 'NOC', 'NOV', 'NOVL',
        'NSC', 'NSM', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVLS', 'NWL', 'NYT', 'NYX',
        'ODP', 'OMC', 'ORCL', 'OXY', 'PAYX', 'PBCT', 'PBG', 'PBI', 'PCAR', 'PCG',
        'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGN', 'PGR',
        'PH', 'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNW', 'POM', 'PPG',
        'PPL', 'PRU', 'PSA', 'PTV', 'LIN', 'PXD', 'Q', 'QCOM', 'QLGC', 'R',
        'RAI', 'RDC', 'RF', 'RHI', 'RL', 'ROH', 'ROK', 'RRC', 'RRD', 'RSG',
        'RSHCQ', 'RTN', 'RX', 'S', 'SBUX', 'SCHW', 'SE', 'SEE', 'SGP', 'SHLD',
        'SHW', 'SIAL', 'SII', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI', 'SO',
        'SOV', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STR', 'STT',
        'STZ', 'SUN', 'SUNEQ', 'SVU', 'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY',
        'T', 'TAP', 'TDC', 'TE', 'TEG', 'TEL', 'TER', 'TGNA', 'TGT', 'THC',
        'TIE', 'TIF', 'TJX', 'TLAB', 'GL', 'TMO', 'TPR', 'TROW', 'TRV', 'TSN',
        'TSS', 'TWX', 'TXN', 'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'USB', 'UST',
        'RTX', 'VAR', 'VFC', 'VIAB', 'VIAV', 'VLO', 'VMC', 'VNO', 'VRSN', 'VZ',
        'WAT', 'WB', 'WBA', 'WEC', 'WFC', 'WFM', 'WFT', 'WHR', 'WIN', 'WM',
        'WMB', 'WMT', 'WU', 'WY', 'WYE', 'TNL', 'WYNN', 'X', 'XEL', 'XL',
        'XLNX', 'XOM', 'XRAY', 'XRX', 'XTO', 'YUM', 'ZBH', 'ZION',
    ])),
    2009: list(set([  # snapshot as of 2009-12-30, 499 tickers
        'A', 'AABA', 'AAPL', 'COR', 'ABT', 'ACS', 'ADBE', 'ADI', 'ADM', 'ADP',
        'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ',
        'AKAM', 'AKS', 'ALL', 'ALTR', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT', 'AMZN',
        'AN', 'ANDV', 'ANF', 'ELV', 'AON', 'APA', 'APC', 'APD', 'APH', 'APOL',
        'ARG', 'HWM', 'ATGE', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AYE', 'AZO',
        'BA', 'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDK', 'BDX', 'BEAM',
        'BEN', 'BF.B', 'BKR', 'BIG', 'BIIB', 'BJS', 'BNY', 'BKNG', 'BALL', 'BMC',
        'BMS', 'BMY', 'BNI', 'BRCM', 'BSX', 'BTUUQ', 'BXP', 'C', 'CA', 'CAG',
        'CAH', 'CAM', 'CAT', 'CB', 'CBRE', 'PSKY', 'CCE', 'CCL', 'CEG', 'CELG',
        'CEPH', 'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX',
        'CMA', 'CMCSA', 'CME', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL',
        'COP', 'COST', 'CPB', 'CPWR', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH',
        'CTXS', 'CVH', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DELL', 'DF', 'DFS',
        'DGX', 'DHI', 'DHR', 'DIS', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DRI',
        'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'EBAY', 'ECL', 'ED',
        'EFX', 'EIX', 'EKDKQ', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EP', 'EQR',
        'EQT', 'ES', 'ESRX', 'ETFC', 'ETN', 'ETR', 'EXC', 'EXPD', 'EXPE', 'F',
        'FAST', 'FCX', 'FDO', 'FDX', 'FE', 'FHN', 'FHI', 'FIS', 'FISV', 'FITB',
        'FLIR', 'FLR', 'FLS', 'FMC', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS',
        'GD', 'GE', 'GENZ', 'GHC', 'GILD', 'GIS', 'GLW', 'GME', 'GNW', 'GOOGL',
        'GPC', 'GAP', 'GR', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN',
        'HCBK', 'DOC', 'HD', 'HES', 'HIG', 'HNZ', 'HOG', 'HON', 'HOT', 'HPQ',
        'HRB', 'HRL', 'LHX', 'HSH', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE',
        'IFF', 'IGT', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'ITT', 'ITW',
        'IVZ', 'JAVA', 'JBL', 'JCI', 'JCP', 'J', 'JEF', 'JNJ', 'JNPR', 'JNS',
        'JPM', 'JWN', 'K', 'KDP', 'KEY', 'KG', 'KIM', 'KLAC', 'KMB', 'KO',
        'KR', 'KSS', 'L', 'BBWI', 'LDOS', 'LEG', 'LEN', 'LH', 'LIFE', 'LLL',
        'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LO', 'LOW', 'LSI', 'LUV', 'LXK',
        'M', 'MA', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ',
        'MDP', 'MDT', 'MEE', 'MET', 'MFE', 'MHS', 'MI', 'MIL', 'MJN', 'MKC',
        'MRSH', 'MMM', 'MO', 'MOLX', 'MON', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI',
        'MTB', 'MU', 'MUR', 'MWV', 'MWW', 'VTRS', 'NBL', 'NBR', 'NDAQ', 'NEE',
        'NEM', 'NI', 'NKE', 'NOC', 'NOV', 'NOVL', 'NSC', 'NSM', 'NTAP', 'NTRS',
        'NUE', 'NVDA', 'NVLS', 'NWL', 'NYT', 'NYX', 'ODP', 'OI', 'OMC', 'ORCL',
        'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBG', 'PBI', 'PCAR', 'PCG', 'PCL', 'PCP',
        'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGN', 'PGR', 'PH', 'PHM',
        'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNW', 'POM', 'PPG', 'PPL', 'PRU',
        'PSA', 'PTV', 'PWR', 'LIN', 'PXD', 'Q', 'QCOM', 'QLGC', 'R', 'RAI',
        'RDC', 'RF', 'RHI', 'RHT', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RRD',
        'RSG', 'RSHCQ', 'RTN', 'RX', 'S', 'SBUX', 'SCG', 'SCHW', 'SE', 'SEE',
        'SHLD', 'SHW', 'SIAL', 'SII', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI',
        'SO', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STR', 'STT',
        'STZ', 'SUN', 'SUNEQ', 'SVU', 'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY',
        'T', 'TAP', 'TDC', 'TE', 'TEG', 'TER', 'TGNA', 'TGT', 'THC', 'TIE',
        'TIF', 'TJX', 'TLAB', 'GL', 'TMO', 'TMUS', 'TPR', 'TROW', 'TRV', 'TSN',
        'TSS', 'TWC', 'TWX', 'TXN', 'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'USB',
        'RTX', 'V', 'VAR', 'VFC', 'VIAB', 'VIAV', 'VLO', 'VMC', 'VNO', 'VRSN',
        'VTR', 'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR',
        'WIN', 'WM', 'WMB', 'WMT', 'WU', 'WY', 'TNL', 'WYNN', 'X', 'XEL',
        'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XTO', 'YUM', 'ZBH', 'ZION',
    ])),
    2010: list(set([  # snapshot as of 2010-12-31, 497 tickers
        'A', 'AABA', 'AAPL', 'COR', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK',
        'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM',
        'AKS', 'ALL', 'ALTR', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN',
        'ANDV', 'ANF', 'ELV', 'AON', 'APA', 'APC', 'APD', 'APH', 'APOL', 'ARG',
        'HWM', 'ATGE', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AYE', 'AZO', 'BA',
        'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDX', 'BEAM', 'BEN', 'BF.B',
        'BKR', 'BIG', 'BIIB', 'BNY', 'BKNG', 'BALL', 'BMC', 'BMS', 'BMY', 'BRCM',
        'BRK.B', 'BSX', 'BTUUQ', 'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM', 'CAT',
        'CB', 'CBRE', 'PSKY', 'CCE', 'CCL', 'CEG', 'CELG', 'CEPH', 'CERN', 'CF',
        'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX', 'CMA', 'CMCSA',
        'CME', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL', 'COP', 'COST',
        'CPB', 'CPWR', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH', 'CTXS', 'CVC',
        'CVH', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DELL', 'DF', 'DFS', 'DGX',
        'DHI', 'DHR', 'DIS', 'WBD', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DRI',
        'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'EBAY', 'ECL', 'ED',
        'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EP', 'EQR', 'EQT',
        'ES', 'ESRX', 'ETFC', 'ETN', 'ETR', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST',
        'FCX', 'FDO', 'FDX', 'FE', 'FFIV', 'FHN', 'FHI', 'FIS', 'FISV', 'FITB',
        'FLIR', 'FLR', 'FLS', 'FMC', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS',
        'GD', 'GE', 'GENZ', 'GHC', 'GILD', 'GIS', 'GLW', 'GME', 'GNW', 'GOOGL',
        'GPC', 'GAP', 'GR', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN',
        'HCBK', 'DOC', 'HD', 'HES', 'HIG', 'HNZ', 'HOG', 'HON', 'HOT', 'HP',
        'HPQ', 'HRB', 'HRL', 'LHX', 'HSH', 'HSP', 'HST', 'HSY', 'HUM', 'IBM',
        'ICE', 'IFF', 'IGT', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG',
        'ITT', 'ITW', 'IVZ', 'JBL', 'JCI', 'JCP', 'J', 'JEF', 'JNJ', 'JNPR',
        'JNS', 'JPM', 'JWN', 'K', 'KDP', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX',
        'KO', 'KR', 'KSS', 'L', 'BBWI', 'LDOS', 'LEG', 'LEN', 'LH', 'LIFE',
        'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LO', 'LOW', 'LSI', 'LUV',
        'LXK', 'M', 'MA', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO',
        'MDLZ', 'MDP', 'MDT', 'MEE', 'MET', 'MFE', 'MHS', 'MI', 'MJN', 'MKC',
        'MRSH', 'MMM', 'MO', 'MOLX', 'MON', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI',
        'MTB', 'MU', 'MUR', 'MWV', 'MWW', 'VTRS', 'NBL', 'NBR', 'NDAQ', 'NEE',
        'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NOC', 'NOV', 'NOVL', 'NRG', 'NSC',
        'NSM', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVLS', 'NWL', 'NYX', 'OI', 'OKE',
        'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG', 'PCL',
        'PCP', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGN', 'PGR', 'PH',
        'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNW', 'POM', 'PPG', 'PPL',
        'PRU', 'PSA', 'PWR', 'LIN', 'PXD', 'Q', 'QCOM', 'QEP', 'QLGC', 'R',
        'RAI', 'RDC', 'RF', 'RHI', 'RHT', 'RL', 'ROK', 'ROP', 'ROST', 'RRC',
        'RRD', 'RSG', 'RSHCQ', 'RTN', 'S', 'SBUX', 'SCG', 'SCHW', 'SE', 'SEE',
        'SHLD', 'SHW', 'SIAL', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI', 'SO',
        'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STZ', 'SUN',
        'SUNEQ', 'SVU', 'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY', 'T', 'TAP',
        'TDC', 'TE', 'TEG', 'TER', 'TGNA', 'TGT', 'THC', 'TIE', 'TIF', 'TJX',
        'TLAB', 'GL', 'TMO', 'TMUS', 'TPR', 'TROW', 'TRV', 'TSN', 'TSS', 'TWC',
        'TWX', 'TXN', 'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'RTX',
        'V', 'VAR', 'VFC', 'VIAB', 'VIAV', 'VLO', 'VMC', 'VNO', 'VRSN', 'VTR',
        'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WIN',
        'WM', 'WMB', 'WMT', 'WU', 'WY', 'TNL', 'WYNN', 'X', 'XEL', 'XL',
        'XLNX', 'XOM', 'XRAY', 'XRX', 'YUM', 'ZBH', 'ZION',
    ])),
    2011: list(set([  # snapshot as of 2011-12-30, 497 tickers
        'A', 'AABA', 'AAPL', 'COR', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP',
        'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ',
        'AKAM', 'ALL', 'ALTR', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN',
        'ANDV', 'ANF', 'ANRZQ', 'ELV', 'AON', 'APA', 'APC', 'APD', 'APH', 'APOL',
        'ARG', 'HWM', 'ATGE', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AZO', 'BA',
        'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDX', 'BEAM', 'BEN', 'BF.B',
        'BKR', 'BIG', 'BIIB', 'BNY', 'BKNG', 'BLK', 'BALL', 'BMC', 'BMS', 'BMY',
        'BRCM', 'BRK.B', 'BSX', 'BTUUQ', 'BWA', 'BXP', 'C', 'CA', 'CAG', 'CAH',
        'CAM', 'CAT', 'CB', 'CBE', 'CBRE', 'PSKY', 'CCE', 'CCL', 'CEG', 'CELG',
        'CERN', 'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX',
        'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA',
        'COL', 'COP', 'COST', 'COV', 'CPB', 'CPWR', 'CRM', 'CSCO', 'CSX', 'CTAS',
        'LUMN', 'CTSH', 'CTXS', 'CVC', 'CVH', 'CVS', 'CVX', 'D', 'DD', 'DE',
        'DELL', 'DF', 'DFS', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DLTR', 'DNB',
        'DNR', 'DO', 'DOV', 'DOW', 'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN',
        'DXC', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN',
        'EMR', 'EOG', 'EP', 'EQR', 'EQT', 'ES', 'ESRX', 'ETFC', 'ETN', 'ETR',
        'EW', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE',
        'FFIV', 'FHN', 'FHI', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC',
        'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS', 'GD', 'GE', 'GHC', 'GILD',
        'GIS', 'GLW', 'GME', 'GNW', 'GOOGL', 'GPC', 'GAP', 'GR', 'GS', 'GT',
        'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'DOC', 'HD', 'HES', 'HIG',
        'HNZ', 'HOG', 'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSH',
        'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU',
        'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JBL', 'JCI', 'JCP',
        'J', 'JEF', 'JNJ', 'JNPR', 'JOY', 'JPM', 'JWN', 'K', 'KDP', 'KEY',
        'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR', 'KSS', 'L', 'BBWI', 'LDOS',
        'LEG', 'LEN', 'LH', 'LIFE', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC',
        'LO', 'LOW', 'LSI', 'LUV', 'LXK', 'M', 'MA', 'MAR', 'MAS', 'MAT',
        'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MHS', 'MJN', 'MKC',
        'MRSH', 'MMI', 'MMM', 'MO', 'MOLX', 'MON', 'MOS', 'MPC', 'MRK', 'MRO',
        'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'MWV', 'VTRS', 'NBL', 'NBR',
        'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NOC', 'NOV',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVLS', 'NWL', 'NYX', 'OI',
        'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG',
        'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGN', 'PGR',
        'PH', 'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNW', 'POM', 'PPG',
        'PPL', 'PRGO', 'PRU', 'PSA', 'PWR', 'LIN', 'PXD', 'QCOM', 'QEP', 'R',
        'RAI', 'RDC', 'RF', 'RHI', 'RHT', 'RL', 'ROK', 'ROP', 'ROST', 'RRC',
        'RRD', 'RSG', 'RTN', 'S', 'SBUX', 'SCG', 'SCHW', 'SE', 'SEE', 'SHLD',
        'SHW', 'SIAL', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI', 'SO', 'SPG',
        'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STZ', 'SUN', 'SVU',
        'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDC', 'TE',
        'TEG', 'TEL', 'TER', 'TGNA', 'TGT', 'THC', 'TIE', 'TIF', 'TJX', 'GL',
        'TMO', 'TMUS', 'TPR', 'TRIP', 'TROW', 'TRV', 'TSN', 'TSS', 'TWC', 'TWX',
        'TXN', 'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'RTX', 'V',
        'VAR', 'VFC', 'VIAB', 'VIAV', 'VLO', 'VMC', 'VNO', 'VRSN', 'VTR', 'VZ',
        'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WIN', 'WM',
        'WMB', 'WMT', 'WU', 'WY', 'TNL', 'WYNN', 'X', 'XEL', 'XL', 'XLNX',
        'XOM', 'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH', 'ZION',
    ])),
    2012: list(set([  # snapshot as of 2012-12-28, 497 tickers
        'A', 'AABA', 'AAPL', 'COR', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP',
        'ADSK', 'ADT', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV',
        'AIZ', 'AKAM', 'ALL', 'ALTR', 'ALXN', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT',
        'AMZN', 'AN', 'ANDV', 'ANF', 'ELV', 'AON', 'APA', 'APC', 'APD', 'APH',
        'APOL', 'APTV', 'ARG', 'HWM', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP', 'AZO',
        'BA', 'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDX', 'BEAM', 'BEN',
        'BF.B', 'BKR', 'BIG', 'BIIB', 'BNY', 'BKNG', 'BLK', 'BALL', 'BMC', 'BMS',
        'BMY', 'BRCM', 'BRK.B', 'BSX', 'BTUUQ', 'BWA', 'BXP', 'C', 'CA', 'CAG',
        'CAH', 'CAM', 'CAT', 'CB', 'CBRE', 'PSKY', 'CCE', 'CCI', 'CCL', 'CELG',
        'CERN', 'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX',
        'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA',
        'COL', 'COP', 'COST', 'COV', 'CPB', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN',
        'CTSH', 'CTXS', 'CVC', 'CVH', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DELL',
        'DF', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DLTR', 'DNB',
        'DNR', 'DO', 'DOV', 'DOW', 'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN',
        'DXC', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN',
        'EMR', 'EOG', 'EQR', 'EQT', 'ES', 'ESRX', 'ESV', 'ETFC', 'ETN', 'ETR',
        'EW', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE',
        'FFIV', 'FHN', 'FHI', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC',
        'FOSL', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS', 'GD', 'GE', 'GHC',
        'GILD', 'GIS', 'GLW', 'GME', 'GNW', 'GOOGL', 'GPC', 'GAP', 'GRMN', 'GS',
        'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'DOC', 'HD', 'HES',
        'HIG', 'HNZ', 'HOG', 'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'LHX',
        'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU',
        'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JBL', 'JCI', 'JCP',
        'J', 'JEF', 'JNJ', 'JNPR', 'JOY', 'JPM', 'JWN', 'K', 'KDP', 'KEY',
        'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KRFT', 'KSS', 'L',
        'BBWI', 'LDOS', 'LEG', 'LEN', 'LH', 'LIFE', 'LLL', 'LLTC', 'LLY', 'LM',
        'LMT', 'LNC', 'LO', 'LOW', 'LRCX', 'LSI', 'LUV', 'LYB', 'M', 'MA',
        'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET',
        'MJN', 'MKC', 'MRSH', 'MMM', 'MNST', 'MO', 'MOLX', 'MON', 'MOS', 'MPC',
        'MRK', 'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'MWV', 'VTRS',
        'NBL', 'NBR', 'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE',
        'NOC', 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NYX',
        'OI', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR',
        'PCG', 'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PETM', 'PFE', 'PFG', 'PG',
        'PGR', 'PH', 'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNR', 'PNW',
        'POM', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PWR', 'LIN', 'PXD',
        'QCOM', 'QEP', 'R', 'RAI', 'RDC', 'RF', 'RHI', 'RHT', 'RL', 'ROK',
        'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'S', 'SBUX', 'SCG', 'SCHW', 'SE',
        'SEE', 'SHW', 'SIAL', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI', 'SO',
        'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STX', 'STZ',
        'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDC', 'TE',
        'TEG', 'TEL', 'TER', 'TGNA', 'TGT', 'THC', 'TIF', 'TJX', 'GL', 'TMO',
        'TMUS', 'TPR', 'TRIP', 'TROW', 'TRV', 'TSN', 'TSS', 'TWC', 'TWX', 'TXN',
        'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'RTX', 'V', 'VAR',
        'VFC', 'VIAB', 'VIAV', 'VLO', 'VMC', 'VNO', 'VRSN', 'VTR', 'VZ', 'WAT',
        'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WIN', 'WM', 'WMB',
        'WMT', 'WPX', 'WU', 'WY', 'TNL', 'WYNN', 'X', 'XEL', 'XL', 'XLNX',
        'XOM', 'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH', 'ZION',
    ])),
    2013: list(set([  # snapshot as of 2013-12-31, 497 tickers
        'A', 'AABA', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'BFH', 'ADSK', 'ADT', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN',
        'AIG', 'AIV', 'AIZ', 'AKAM', 'ALL', 'ALLE', 'ALTR', 'ALXN', 'AMAT', 'AME',
        'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'ANDV', 'ELV', 'AON', 'APA', 'APC',
        'APD', 'APH', 'APTV', 'ARG', 'HWM', 'ATI', 'AVB', 'AVP', 'AVY', 'AXP',
        'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR', 'BDX', 'BEAM',
        'BEN', 'BF.B', 'BKR', 'BIIB', 'BNY', 'BKNG', 'BLK', 'BALL', 'BMS', 'BMY',
        'BRCM', 'BRK.B', 'BSX', 'BTUUQ', 'BWA', 'BXP', 'C', 'CA', 'CAG', 'CAH',
        'CAM', 'CAT', 'CB', 'CBRE', 'PSKY', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN',
        'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL',
        'COP', 'COST', 'COV', 'CPB', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH',
        'CTXS', 'CVC', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG',
        'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DLTR', 'DNB', 'DNR', 'DO', 'DOV',
        'DOW', 'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'EBAY',
        'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EQR',
        'EQT', 'ES', 'ESRX', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC', 'EXPD',
        'EXPE', 'F', 'FAST', 'META', 'FCX', 'FDO', 'FDX', 'FE', 'FFIV', 'FIS',
        'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FOXA', 'FRX', 'FSLR',
        'FTI', 'FTR', 'GAS', 'GD', 'GE', 'GGP', 'GHC', 'GILD', 'GIS', 'GLW',
        'GM', 'GME', 'GNW', 'GOOGL', 'GPC', 'GAP', 'GRMN', 'GS', 'GT', 'GWW',
        'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'DOC', 'HD', 'HES', 'HIG', 'HOG',
        'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSP', 'HST', 'HSY',
        'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU', 'IP', 'IPG', 'IR',
        'IRM', 'ISRG', 'ITW', 'IVZ', 'JBL', 'JCI', 'J', 'JEF', 'JNJ', 'JNPR',
        'JOY', 'JPM', 'JWN', 'K', 'KDP', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMI',
        'KMX', 'KO', 'CPRI', 'KR', 'KRFT', 'KSS', 'KSU', 'L', 'BBWI', 'LEG',
        'LEN', 'LH', 'LIFE', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LO',
        'LOW', 'LRCX', 'LSI', 'LUV', 'LYB', 'M', 'MA', 'MAC', 'MAR', 'MAS',
        'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MHK', 'MJN',
        'MKC', 'MRSH', 'MMM', 'MNST', 'MO', 'MON', 'MOS', 'MPC', 'MRK', 'MRO',
        'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'MWV', 'VTRS', 'NBL', 'NBR',
        'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC',
        'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWSA', 'OI',
        'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG',
        'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PETM', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNR', 'PNW', 'POM',
        'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'LIN', 'PXD',
        'QCOM', 'QEP', 'R', 'RAI', 'RDC', 'REGN', 'RF', 'RHI', 'RHT', 'RIG',
        'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBUX', 'SCG', 'SCHW',
        'SE', 'SEE', 'SHW', 'SIAL', 'SJM', 'SLB', 'SLM', 'SNA', 'SNDK', 'SNI',
        'SO', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STX',
        'STZ', 'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDC',
        'TE', 'TEG', 'TEL', 'TGNA', 'TGT', 'THC', 'TIF', 'TJX', 'GL', 'TMO',
        'TPR', 'TRIP', 'TROW', 'TRV', 'TSN', 'TSS', 'TWC', 'TWX', 'TXN', 'TXT',
        'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'RTX', 'V', 'VAR', 'VFC',
        'VIAB', 'VLO', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA',
        'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WIN', 'WM', 'WMB', 'WMT',
        'WPX', 'WU', 'WY', 'TNL', 'WYNN', 'X', 'XEL', 'XL', 'XLNX', 'XOM',
        'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS',
    ])),
    2014: list(set([  # snapshot as of 2014-12-24, 499 tickers
        'A', 'AABA', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'BFH', 'ADSK', 'ADT', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN',
        'AIG', 'AIV', 'AIZ', 'AKAM', 'ALL', 'ALLE', 'ALTR', 'ALXN', 'AMAT', 'AME',
        'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'ANDV', 'ELV', 'AON', 'APA',
        'APC', 'APD', 'APH', 'APTV', 'ARG', 'HWM', 'ATI', 'AVB', 'AVGO', 'AVP',
        'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR',
        'BDX', 'BEN', 'BF.B', 'BKR', 'BIIB', 'BNY', 'BKNG', 'BLK', 'BALL', 'BMY',
        'BRCM', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM',
        'CAT', 'CB', 'CBRE', 'PSKY', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN', 'CF',
        'CFN', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME',
        'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL', 'COP', 'COST',
        'COV', 'CPB', 'CRM', 'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH', 'CTXS', 'CVC',
        'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI',
        'DHR', 'DIS', 'WBD', 'DISCK', 'DLTR', 'DNB', 'DNR', 'DO', 'DOV', 'DOW',
        'DRI', 'DTE', 'DTV', 'DUK', 'DVA', 'DVN', 'DXC', 'EA', 'EBAY', 'ECL',
        'ED', 'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EQR', 'EQT',
        'ES', 'ESRX', 'ESS', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC', 'EXPD',
        'EXPE', 'F', 'FAST', 'META', 'FCX', 'FDO', 'FDX', 'FE', 'FFIV', 'FIS',
        'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FOXA', 'FSLR', 'FTI',
        'FTR', 'GAS', 'GD', 'GE', 'GGP', 'GILD', 'GIS', 'GLW', 'GM', 'GMCR',
        'GME', 'GNW', 'GOOG', 'GOOGL', 'GPC', 'GAP', 'GRMN', 'GS', 'GT', 'GWW',
        'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'DOC', 'HD', 'HES', 'HIG', 'HOG',
        'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSP', 'HST', 'HSY',
        'HUM', 'IBM', 'ICE', 'IFF', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM',
        'ISRG', 'ITW', 'IVZ', 'JCI', 'J', 'JEF', 'JNJ', 'JNPR', 'JOY', 'JPM',
        'JWN', 'K', 'KDP', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO',
        'CPRI', 'KR', 'KRFT', 'KSS', 'KSU', 'L', 'BBWI', 'LEG', 'LEN', 'LH',
        'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LO', 'LOW', 'LRCX', 'LUV',
        'LVLT', 'LYB', 'M', 'MA', 'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP',
        'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MHK', 'MJN', 'MKC', 'MLM', 'MRSH',
        'MMM', 'MNK', 'MNST', 'MO', 'MON', 'MOS', 'MPC', 'MRK', 'MRO', 'MS',
        'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'MWV', 'VTRS', 'NAVI', 'NBL', 'NBR',
        'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC',
        'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWSA', 'OI',
        'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG',
        'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PETM', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'RVTY', 'PLD', 'PLL', 'PM', 'PNC', 'PNR', 'PNW', 'POM',
        'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'LIN', 'PXD',
        'QCOM', 'QEP', 'R', 'RAI', 'RCL', 'REGN', 'RF', 'RHI', 'RHT', 'RIG',
        'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBUX', 'SCG', 'SCHW',
        'SE', 'SEE', 'SHW', 'SIAL', 'SJM', 'SLB', 'SNA', 'SNDK', 'SNI', 'SO',
        'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STX', 'STZ',
        'SWK', 'SWN', 'SWY', 'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDC', 'TE',
        'TEG', 'TEL', 'TGNA', 'TGT', 'THC', 'TIF', 'TJX', 'GL', 'TMO', 'TPR',
        'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWC', 'TWX', 'TXN', 'TXT',
        'UAA', 'UHS', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'URI', 'USB', 'RTX',
        'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VTR',
        'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WIN',
        'WM', 'WMB', 'WMT', 'WU', 'WY', 'TNL', 'WYNN', 'XEC', 'XEL', 'XL',
        'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS',
    ])),
    2015: list(set([  # snapshot as of 2015-12-31, 502 tickers
        'A', 'AABA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'BFH', 'ADSK', 'ADT', 'AEE', 'AEP', 'AES', 'AET',
        'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM', 'ALL', 'ALLE', 'ALXN', 'AMAT',
        'AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'ANDV', 'ELV', 'AON',
        'APA', 'APC', 'APD', 'APH', 'APTV', 'ARG', 'HWM', 'ATVI', 'AVB', 'AVGO',
        'AVY', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'TFC', 'BBY', 'BCR',
        'BDX', 'BEN', 'BF.B', 'BKR', 'BIIB', 'BNY', 'BKNG', 'BLK', 'BALL', 'BMY',
        'BRCM', 'BRK.B', 'BSX', 'BWA', 'BXLT', 'BXP', 'C', 'CA', 'CAG', 'CAH',
        'CAM', 'CAT', 'CB', 'CBRE', 'PSKY', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN',
        'CF', 'CHD', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA',
        'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'CTRA', 'COL', 'COP',
        'COST', 'CPB', 'CPGX', 'CRM', 'CSCO', 'CSRA', 'CSX', 'CTAS', 'LUMN', 'CTSH',
        'CTXS', 'CVC', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG',
        'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DISCK', 'DLTR', 'DNB', 'DO', 'DOV',
        'DOW', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED',
        'EFX', 'EIX', 'EL', 'EMC', 'EMN', 'EMR', 'ENDP', 'EOG', 'EQIX', 'EQR',
        'EQT', 'ES', 'ESRX', 'ESS', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC',
        'EXPD', 'EXPE', 'F', 'FAST', 'META', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS',
        'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FOX', 'FOXA', 'FSLR',
        'FTI', 'FTR', 'GAS', 'GD', 'GE', 'GGP', 'GILD', 'GIS', 'GLW', 'GM',
        'GMCR', 'GME', 'GOOG', 'GOOGL', 'GPC', 'GAP', 'GRMN', 'GS', 'GT', 'GWW',
        'HAL', 'HAR', 'HAS', 'HBAN', 'HBI', 'HCA', 'DOC', 'HD', 'HES', 'HIG',
        'HOG', 'HON', 'HOT', 'HP', 'HPE', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSIC',
        'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'ILMN', 'INTC', 'INTU', 'IP',
        'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JBHT', 'JCI', 'J', 'JEF',
        'JNJ', 'JNPR', 'JPM', 'JWN', 'K', 'KDP', 'KEY', 'KHC', 'KIM', 'KLAC',
        'KMB', 'KMI', 'KMX', 'KO', 'CPRI', 'KR', 'KSS', 'KSU', 'L', 'BBWI',
        'LEG', 'LEN', 'LH', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LOW',
        'LRCX', 'LUV', 'LVLT', 'LYB', 'M', 'MA', 'MAC', 'MAR', 'MAS', 'MAT',
        'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MHK', 'MJN', 'MKC',
        'MLM', 'MRSH', 'MMM', 'MNK', 'MNST', 'MO', 'MON', 'MOS', 'MPC', 'MRK',
        'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'VTRS', 'NAVI', 'NBL',
        'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC', 'NOV',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'O',
        'OI', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR',
        'PCG', 'PCL', 'PCP', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POM', 'PPG',
        'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'LIN', 'PXD', 'PYPL',
        'QCOM', 'QRVO', 'R', 'RAI', 'RCL', 'REGN', 'RF', 'RHI', 'RHT', 'RIG',
        'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBUX', 'SCG', 'SCHW',
        'SE', 'SEE', 'SHW', 'SIG', 'SJM', 'SLB', 'SLG', 'SNA', 'SNDK', 'SNI',
        'SO', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STX',
        'STZ', 'SWK', 'SWKS', 'SWN', 'SYF', 'SYK', 'GEN', 'SYY', 'T', 'TAP',
        'TDC', 'TE', 'TEL', 'TGNA', 'TGT', 'THC', 'TIF', 'TJX', 'GL', 'TMO',
        'TPR', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWC', 'TWX', 'TXN',
        'TXT', 'UAA', 'UAL', 'UHS', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'URI',
        'USB', 'RTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSK',
        'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WELL', 'WFC',
        'WFM', 'WHR', 'WM', 'WMB', 'WMT', 'WRK', 'WU', 'WY', 'TNL', 'WYNN',
        'XEC', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH',
        'ZION', 'ZTS',
    ])),
    2016: list(set([  # snapshot as of 2016-12-30, 506 tickers
        'A', 'AABA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'BFH', 'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL',
        'AGN', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALK', 'ALL', 'ALLE',
        'ALXN', 'AMAT', 'AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN', 'ANDV',
        'ELV', 'AON', 'APA', 'APC', 'APD', 'APH', 'APTV', 'HWM', 'ATVI', 'AVB',
        'AVGO', 'AVY', 'AWK', 'AXP', 'AYI', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY',
        'TFC', 'BBY', 'BCR', 'BDX', 'BEN', 'BF.B', 'BKR', 'BIIB', 'BNY', 'BKNG',
        'BLK', 'BALL', 'BMY', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CA', 'CAG',
        'CAH', 'CAT', 'CB', 'CBRE', 'PSKY', 'CCI', 'CCL', 'CELG', 'CERN', 'CF',
        'CFG', 'CHD', 'CHK', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'CTRA', 'COL',
        'COO', 'COP', 'COST', 'COTY', 'CPB', 'CPRI', 'CRM', 'CSCO', 'CSRA', 'CSX',
        'CTAS', 'LUMN', 'CTSH', 'CTXS', 'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DD',
        'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DISCK', 'DLR',
        'DLTR', 'DNB', 'DOV', 'DOW', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'EA',
        'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENDP', 'EOG',
        'EQIX', 'EQR', 'EQT', 'ES', 'ESRX', 'ESS', 'ETFC', 'ETN', 'ETR', 'EVHC',
        'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FAST', 'META', 'FBIN', 'FCX',
        'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FL', 'FLIR', 'FLR', 'FLS',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTI', 'FTR', 'FTV', 'GD', 'GE',
        'GGP', 'GILD', 'GIS', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GAP',
        'GRMN', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HBI', 'HCA',
        'DOC', 'HD', 'HES', 'HIG', 'HOG', 'HOLX', 'HON', 'HP', 'HPE', 'HPQ',
        'HRB', 'HRL', 'LHX', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF',
        'ILMN', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ',
        'JBHT', 'JCI', 'J', 'JEF', 'JNJ', 'JNPR', 'JPM', 'JWN', 'K', 'KDP',
        'KEY', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR',
        'KSS', 'KSU', 'L', 'BBWI', 'LEG', 'LEN', 'LH', 'LKQ', 'LLL', 'LLTC',
        'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVLT', 'LYB', 'M',
        'MA', 'MAA', 'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO',
        'MDLZ', 'MDT', 'MET', 'MHK', 'MJN', 'MKC', 'MLM', 'MRSH', 'MMM', 'MNK',
        'MNST', 'MO', 'MON', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI',
        'MTB', 'MTD', 'MU', 'MUR', 'VTRS', 'NAVI', 'NBL', 'NDAQ', 'NEE', 'NEM',
        'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC', 'NOV', 'NRG', 'NSC', 'NTAP',
        'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'O', 'OKE', 'OMC', 'ORCL',
        'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG', 'PDCO', 'PEG', 'PEP',
        'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'RVTY', 'PLD', 'PM', 'PNC',
        'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR',
        'LIN', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'R', 'RAI', 'RCL', 'REGN', 'RF',
        'RHI', 'RHT', 'RIG', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN',
        'SBUX', 'SCG', 'SCHW', 'SE', 'SEE', 'SHW', 'SIG', 'SJM', 'SLB', 'SLG',
        'SNA', 'SNI', 'SO', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ',
        'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SWN', 'SYF', 'SYK', 'GEN', 'SYY',
        'T', 'TAP', 'TDC', 'TDG', 'TEL', 'TGNA', 'TGT', 'TIF', 'TJX', 'GL',
        'TMO', 'TPR', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWX', 'TXN',
        'TXT', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP',
        'UPS', 'URBN', 'URI', 'USB', 'RTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO',
        'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA', 'WDC',
        'WEC', 'WELL', 'WFC', 'WFM', 'WHR', 'WTW', 'WM', 'WMB', 'WMT', 'WRK',
        'WU', 'WY', 'TNL', 'WYNN', 'XEC', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY',
        'XRX', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS',
    ])),
    2017: list(set([  # snapshot as of 2017-12-29, 505 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACN', 'ADBE', 'ADI',
        'ADM', 'ADP', 'BFH', 'ADSK', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN',
        'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE',
        'ALXN', 'AMAT', 'AMD', 'AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANDV',
        'ANSS', 'ELV', 'AON', 'AOS', 'APA', 'APC', 'APD', 'APH', 'APTV', 'ARE',
        'HWM', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AYI', 'AZO', 'BA',
        'BAC', 'BAX', 'TFC', 'BBY', 'BDX', 'BEN', 'BF.B', 'BHF', 'BKR', 'BIIB',
        'BNY', 'BKNG', 'BLK', 'BALL', 'BMY', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C',
        'CA', 'CAG', 'CAH', 'CAT', 'CB', 'CBOE', 'CBRE', 'PSKY', 'CCI', 'CCL',
        'CDNS', 'CELG', 'CERN', 'CF', 'CFG', 'CHD', 'CHK', 'CHRW', 'CHTR', 'CI',
        'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC',
        'CNP', 'COF', 'CTRA', 'COL', 'COO', 'COP', 'COST', 'COTY', 'CPB', 'CPRI',
        'CRM', 'CSCO', 'CSRA', 'CSX', 'CTAS', 'LUMN', 'CTSH', 'CTXS', 'CVS', 'CVX',
        'CXO', 'D', 'DAL', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
        'WBD', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV', 'DRE', 'DRI', 'DTE', 'DUK',
        'DVA', 'DVN', 'DD', 'DXC', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX',
        'EL', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESRX', 'ESS',
        'ETFC', 'ETN', 'ETR', 'EVHC', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F',
        'FAST', 'META', 'FBIN', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB',
        'FL', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOX', 'FOXA', 'FRT', 'FTI', 'FTV',
        'GD', 'GE', 'GGP', 'GILD', 'GIS', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC',
        'GPN', 'GAP', 'GRMN', 'GS', 'GT', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI',
        'HCA', 'DOC', 'HD', 'HES', 'HIG', 'HLT', 'HOG', 'HOLX', 'HON', 'HP',
        'HPE', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM',
        'ICE', 'IDXX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG',
        'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'JBHT', 'JCI', 'J',
        'JEF', 'JNJ', 'JNPR', 'JPM', 'JWN', 'K', 'KDP', 'KEY', 'KHC', 'KIM',
        'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSS', 'KSU', 'L',
        'BBWI', 'LEG', 'LEN', 'LH', 'LKQ', 'LLL', 'LLY', 'LMT', 'LNC', 'LNT',
        'LOW', 'LRCX', 'LUV', 'LYB', 'M', 'MA', 'MAA', 'MAC', 'MAR', 'MAS',
        'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK',
        'MKC', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO', 'MON', 'MOS', 'MPC', 'MRK',
        'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'VTRS', 'NAVI', 'NBL',
        'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC',
        'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA',
        'O', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PCAR', 'PCG',
        'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG',
        'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU',
        'PSA', 'PSX', 'PVH', 'PWR', 'LIN', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL',
        'EG', 'REG', 'REGN', 'RF', 'RHI', 'RHT', 'RJF', 'RL', 'RMD', 'ROK',
        'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCG', 'SCHW', 'SEE',
        'SHW', 'SIG', 'SJM', 'SLB', 'SLG', 'SNA', 'SNI', 'SNPS', 'SO', 'SPG',
        'SPGI', 'SRCL', 'SRE', 'STI', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF',
        'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDG', 'TEL', 'TGT', 'TIF', 'TJX',
        'GL', 'TMO', 'TPR', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TWX',
        'TXN', 'TXT', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM',
        'UNP', 'UPS', 'URI', 'USB', 'RTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO',
        'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA', 'WDC',
        'WEC', 'WELL', 'WFC', 'WHR', 'WTW', 'WM', 'WMB', 'WMT', 'WRK', 'WU',
        'WY', 'TNL', 'WYNN', 'XEC', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX',
        'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS',
    ])),
    2018: list(set([  # snapshot as of 2018-12-24, 505 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABMD', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'BFH', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGN',
        'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE',
        'ALXN', 'AMAT', 'AMD', 'AME', 'AMG', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET',
        'ANSS', 'ELV', 'AON', 'AOS', 'APA', 'APC', 'APD', 'APH', 'APTV', 'ARE',
        'HWM', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC',
        'BAX', 'TFC', 'BBY', 'BDX', 'BEN', 'BF.B', 'BHF', 'BKR', 'BIIB', 'BNY',
        'BKNG', 'BLK', 'BALL', 'BMY', 'BR', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C',
        'CAG', 'CAH', 'CAT', 'CB', 'CBOE', 'CBRE', 'PSKY', 'CCI', 'CCL', 'CDNS',
        'CE', 'CELG', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
        'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
        'COF', 'CTRA', 'COO', 'COP', 'COST', 'COTY', 'CPB', 'CPRI', 'CPRT', 'CRM',
        'CSCO', 'CSX', 'CTAS', 'LUMN', 'CTSH', 'CTXS', 'CVS', 'CVX', 'CXO', 'D',
        'DAL', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DISCK',
        'DISH', 'DLR', 'DLTR', 'DOV', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
        'DD', 'DXC', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN',
        'EMR', 'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETFC', 'ETN', 'ETR', 'EVRG',
        'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'META', 'FBIN',
        'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FL', 'FLIR', 'FLR',
        'FLS', 'CPAY', 'FMC', 'FOX', 'FOXA', 'FRT', 'FTI', 'FTNT', 'FTV', 'GD',
        'GE', 'GILD', 'GIS', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GAP',
        'GRMN', 'GS', 'GT', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'DOC',
        'HD', 'HES', 'DINO', 'HIG', 'HII', 'HLT', 'HOG', 'HOLX', 'HON', 'HP',
        'HPE', 'HPQ', 'HRB', 'HRL', 'LHX', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM',
        'ICE', 'IDXX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG',
        'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'JBHT', 'JCI',
        'J', 'JEF', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'JWN', 'K', 'KEY', 'KEYS',
        'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSS', 'KSU',
        'L', 'BBWI', 'LEG', 'LEN', 'LH', 'LIN', 'LKQ', 'LLL', 'LLY', 'LMT',
        'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 'LW', 'LYB', 'M', 'MA', 'MAA',
        'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT',
        'MET', 'MGM', 'MHK', 'MKC', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO', 'MOS',
        'MPC', 'MRK', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU',
        'MXIM', 'VTRS', 'NBL', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI',
        'NKE', 'NKTR', 'NLSN', 'NOC', 'NOV', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NWL', 'NWS', 'NWSA', 'O', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY',
        'PAYX', 'PBCT', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'PKG', 'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG',
        'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM',
        'QRVO', 'RCL', 'EG', 'REG', 'REGN', 'RF', 'RHI', 'RHT', 'RJF', 'RL',
        'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCG',
        'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SLG', 'SNA', 'SNPS', 'SO',
        'SPG', 'SPGI', 'SRE', 'STI', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF',
        'SYK', 'GEN', 'SYY', 'T', 'TAP', 'TDG', 'TEL', 'TGT', 'TIF', 'TJX',
        'GL', 'TMO', 'TPR', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSS', 'TTWO',
        'TWTR', 'TXN', 'TXT', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH',
        'UNM', 'UNP', 'UPS', 'URI', 'USB', 'RTX', 'V', 'VAR', 'VFC', 'VIAB',
        'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA',
        'WCG', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WTW', 'WM', 'WMB', 'WMT',
        'WRK', 'WU', 'WY', 'WYNN', 'XEC', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XRX',
        'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS',
    ])),
    2019: list(set([  # snapshot as of 2019-12-23, 505 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABMD', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'BFH', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGN',
        'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE',
        'ALXN', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET',
        'ANSS', 'ELV', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'HWM',
        'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC',
        'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BIIB', 'BNY', 'BKNG', 'BKR', 'BLK',
        'BALL', 'BMY', 'BR', 'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH',
        'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN',
        'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'CTRA', 'COO',
        'COP', 'COST', 'COTY', 'CPB', 'CPRI', 'CPRT', 'CRM', 'CSCO', 'CSX', 'CTAS',
        'LUMN', 'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DD',
        'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DISCK', 'DISH',
        'DLR', 'DLTR', 'DOV', 'DOW', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
        'DXC', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR',
        'EOG', 'EQIX', 'EQR', 'ES', 'ESS', 'ETFC', 'ETN', 'ETR', 'EVRG', 'EW',
        'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'META', 'FBIN', 'FCX',
        'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLS', 'CPAY', 'FMC',
        'FOX', 'FOXA', 'FRC', 'FRT', 'FTI', 'FTNT', 'FTV', 'GD', 'GE', 'GILD',
        'GIS', 'GL', 'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GAP', 'GRMN',
        'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'DINO',
        'HIG', 'HII', 'HLT', 'HOG', 'HOLX', 'HON', 'HP', 'HPE', 'HPQ', 'HRB',
        'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF',
        'ILMN', 'INCY', 'INFO', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR',
        'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ',
        'JNPR', 'JPM', 'JWN', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB',
        'KMI', 'KMX', 'KO', 'KR', 'KSS', 'KSU', 'L', 'BBWI', 'LDOS', 'LEG',
        'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW',
        'LRCX', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'M', 'MA', 'MAA', 'MAR',
        'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK',
        'MKC', 'MKTX', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MRK',
        'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'MXIM', 'VTRS',
        'NBL', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'GEN', 'NLSN',
        'NOC', 'NOV', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
        'NWL', 'NWS', 'NWSA', 'O', 'ODFL', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY',
        'PAYX', 'PBCT', 'PCAR', 'DOC', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'PKG', 'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG',
        'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM',
        'QRVO', 'RCL', 'EG', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD',
        'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTN', 'SBAC', 'SBUX', 'SCHW', 'SEE',
        'SHW', 'SIVB', 'SJM', 'SLB', 'SLG', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI',
        'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY',
        'T', 'TAP', 'TDG', 'TEL', 'TFC', 'TFX', 'TGT', 'TIF', 'TJX', 'TMO',
        'TMUS', 'TPR', 'TROW', 'TRV', 'TSCO', 'TSN', 'TTWO', 'TWTR', 'TXN', 'TXT',
        'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS',
        'URI', 'USB', 'RTX', 'V', 'VAR', 'VFC', 'PSKY', 'VLO', 'VMC', 'VNO',
        'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAB', 'WAT', 'WBA', 'WCG', 'WDC',
        'WEC', 'WELL', 'WFC', 'WHR', 'WTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
        'WU', 'WY', 'WYNN', 'XEC', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL',
        'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS',
    ])),
    2020: list(set([  # snapshot as of 2020-12-21, 505 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABMD', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
        'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'ALXN', 'AMAT', 'AMCR',
        'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ELV', 'AON',
        'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO',
        'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBY', 'BDX', 'BEN',
        'BF.B', 'BIIB', 'BIO', 'BNY', 'BKNG', 'BKR', 'BLK', 'BALL', 'BMY', 'BR',
        'BRK.B', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
        'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN', 'CF', 'CFG',
        'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME',
        'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'CTRA', 'COO', 'COP', 'COST',
        'CPB', 'CPRT', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTSH', 'CTVA', 'CTXS',
        'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX',
        'DHI', 'DHR', 'DIS', 'WBD', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW',
        'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA',
        'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'EOG', 'EQIX',
        'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD',
        'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'META', 'FBIN', 'FCX', 'FDX', 'FE',
        'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLS', 'CPAY', 'FMC', 'FOX', 'FOXA',
        'FRC', 'FRT', 'FTI', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL',
        'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GAP', 'GRMN', 'GS', 'GWW',
        'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'DINO', 'HIG', 'HII',
        'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM',
        'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC',
        'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
        'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY',
        'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSU',
        'L', 'BBWI', 'LDOS', 'LEG', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY',
        'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB',
        'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ',
        'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MRSH', 'MMM', 'MNST',
        'MO', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB',
        'MTD', 'MU', 'MXIM', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE',
        'GEN', 'NLSN', 'NOC', 'NOV', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'O', 'ODFL', 'OKE', 'OMC', 'ORCL',
        'ORLY', 'OTIS', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'DOC', 'PEG', 'PEP',
        'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'RVTY', 'PLD', 'PM',
        'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX',
        'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'EG', 'REG', 'REGN',
        'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
        'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SLG',
        'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ',
        'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TEL',
        'TER', 'TFC', 'TFX', 'TGT', 'TIF', 'TJX', 'TMO', 'TMUS', 'TPR', 'TROW',
        'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL',
        'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS',
        'URI', 'USB', 'V', 'VAR', 'VFC', 'PSKY', 'VLO', 'VMC', 'VNO', 'VNT',
        'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC',
        'WEC', 'WELL', 'WFC', 'WHR', 'WTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
        'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL',
        'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS',
    ])),
    2021: list(set([  # snapshot as of 2021-12-20, 505 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABMD', 'ABT', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
        'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD',
        'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ELV', 'AON', 'AOS',
        'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY',
        'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN',
        'BF.B', 'BIIB', 'BIO', 'BNY', 'BKNG', 'BKR', 'BLK', 'BALL', 'BMY', 'BR',
        'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
        'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'DAY', 'CDNS', 'CDW', 'CE', 'CERN',
        'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP',
        'COST', 'CPB', 'CPRT', 'CRL', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTRA',
        'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE',
        'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'WBD', 'DISCK', 'DISH', 'DLR',
        'DLTR', 'DOV', 'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
        'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN',
        'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR',
        'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST',
        'META', 'FBIN', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB',
        'CPAY', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE',
        'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN',
        'GAP', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES',
        'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST',
        'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY',
        'INFO', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG',
        'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM',
        'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO',
        'KR', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT',
        'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV',
        'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT',
        'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO',
        'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
        'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI',
        'NKE', 'GEN', 'NLSN', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE',
        'OMC', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'DOC',
        'PEG', 'PENN', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG',
        'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU',
        'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL',
        'EG', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL',
        'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEDG', 'SEE',
        'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE',
        'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T',
        'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX',
        'TMO', 'TMUS', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT',
        'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS',
        'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'PSKY', 'VLO',
        'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
        'WBA', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WTW', 'WM', 'WMB', 'WMT',
        'WRB', 'WRK', 'WST', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XYL',
        'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS',
    ])),
    2022: list(set([  # snapshot as of 2022-12-22, 503 tickers
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'COR', 'ABT', 'ACGL', 'ACN', 'ADBE',
        'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
        'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD',
        'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA',
        'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK',
        'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN',
        'BF.B', 'BIIB', 'BIO', 'BNY', 'BKNG', 'BKR', 'BLK', 'BMY', 'BR', 'BRK.B',
        'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
        'CBOE', 'CBRE', 'CCI', 'CCL', 'DAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF',
        'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA',
        'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST',
        'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT',
        'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE',
        'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR', 'DOV',
        'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA',
        'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH',
        'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY',
        'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX',
        'FDS', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'CPAY', 'FMC', 'FOX',
        'FOXA', 'FRC', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEN', 'GILD',
        'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN',
        'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII',
        'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM',
        'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU',
        'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
        'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY',
        'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L',
        'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT',
        'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA',
        'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META',
        'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO', 'MOH',
        'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
        'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX',
        'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA',
        'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC',
        'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PSKY', 'PAYC', 'PAYX', 'PCAR', 'PCG',
        'DOC', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG',
        'RVTY', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU',
        'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'EG',
        'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP',
        'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SEDG', 'SEE', 'SHW',
        'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE',
        'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T',
        'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX',
        'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN',
        'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH',
        'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VNO',
        'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD',
        'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
        'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH',
        'ZBRA', 'ZION', 'ZTS',
    ])),
    2023: list(set([  # snapshot as of 2023-12-18, 503 tickers
        'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI',
        'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG',
        'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN',
        'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH',
        'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO',
        'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG',
        'BIIB', 'BIO', 'BNY', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B',
        'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
        'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'DAY', 'CDNS', 'CDW', 'CE', 'CEG',
        'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
        'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP',
        'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX',
        'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL',
        'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
        'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA',
        'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR',
        'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR',
        'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST',
        'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FISV', 'FICO', 'FIS', 'FITB', 'CPAY',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC',
        'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC',
        'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES',
        'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST',
        'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN',
        'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG',
        'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR',
        'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI',
        'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN',
        'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW',
        'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO',
        'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MRSH',
        'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO',
        'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ',
        'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC',
        'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL',
        'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PSKY', 'PAYC',
        'PAYX', 'PCAR', 'PCG', 'DOC', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR',
        'PH', 'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL',
        'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM',
        'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK',
        'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW',
        'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD',
        'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP',
        'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO',
        'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT',
        'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH',
        'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VLTO', 'VMC',
        'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD',
        'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
        'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH',
        'ZBRA', 'ZION', 'ZTS',
    ])),
    2024: list(set([  # snapshot as of 2024-12-23, 503 tickers
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
        'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP',
        'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO',
        'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO',
        'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB',
        'BNY', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX',
        'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE',
        'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD',
        'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI',
        'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB',
        'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA',
        'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE',
        'DECK', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
        'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN',
        'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS',
        'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG',
        'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FISV', 'FICO', 'FIS', 'FITB',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE',
        'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG',
        'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA',
        'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL',
        'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX',
        'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM',
        'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ',
        'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC',
        'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH',
        'LHX', 'LII', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU',
        'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD',
        'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC',
        'MKTX', 'MLM', 'MRSH', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR',
        'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
        'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI',
        'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW',
        'PSKY', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG',
        'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW',
        'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PYPL',
        'QCOM', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL',
        'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM',
        'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE',
        'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY',
        'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT',
        'TJX', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO',
        'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR',
        'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO',
        'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB',
        'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB',
        'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM',
        'ZBH', 'ZBRA', 'ZTS',
    ])),
    2025: list(set([  # snapshot as of 2025-12-22, 503 tickers
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
        'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP',
        'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP',
        'APTV', 'ARE', 'ARES', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP',
        'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG',
        'BIIB', 'BNY', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO',
        'BSX', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE',
        'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW',
        'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS',
        'CNC', 'CNP', 'COF', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB',
        'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS',
        'CTRA', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DAY',
        'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
        'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
        'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL',
        'ELV', 'EME', 'EMR', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES',
        'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR',
        'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FICO', 'FIS',
        'FISV', 'FITB', 'FIX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD',
        'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
        'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
        'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HOOD', 'HPE',
        'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM',
        'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV',
        'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI',
        'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC',
        'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX',
        'LII', 'LIN', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS',
        'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK',
        'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MRSH', 'MMM',
        'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI',
        'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE',
        'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS',
        'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC',
        'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG',
        'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD',
        'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU',
        'PSA', 'PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'REG',
        'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
        'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA',
        'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
        'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP',
        'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO',
        'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN',
        'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS',
        'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO',
        'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
        'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB',
        'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM',
        'ZBH', 'ZBRA', 'ZTS',
    ])),
    2026: list(set([  # snapshot as of 2026-09-21, 503 tickers
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
        'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
        'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP',
        'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP',
        'APTV', 'ARE', 'ARES', 'ATO', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO',
        'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB',
        'BKNG', 'BKR', 'BLK', 'BMY', 'BNY', 'BR', 'BRK.B', 'BRO', 'BSX',
        'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE',
        'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR',
        'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS',
        'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY',
        'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS',
        'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DDOG',
        'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
        'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM',
        'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME',
        'EMR', 'EOG', 'EQIX', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
        'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX',
        'FDS', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS', 'FISV', 'FITB',
        'FIX', 'FLEX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY',
        'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC',
        'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN',
        'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HON', 'HONA', 'HOOD', 'HPE', 'HPQ',
        'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM', 'ICE',
        'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR',
        'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY',
        'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB',
        'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII',
        'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS',
        'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO',
        'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO',
        'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRVL', 'MS', 'MSCI', 'MSFT',
        'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX',
        'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA',
        'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL',
        'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE',
        'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC',
        'PNR', 'PNW', 'PODD', 'PPG', 'PPL', 'PRU', 'PSA', 'PSKY', 'PSX', 'PTC',
        'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF',
        'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC',
        'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNDK', 'SNPS', 'SO',
        'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW',
        'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TDG', 'TDY', 'TECH',
        'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL', 'TPR',
        'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO',
        'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP',
        'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VLTO', 'VMC', 'VMRK',
        'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
        'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB',
        'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM',
        'ZBH', 'ZBRA', 'ZTS',
        'BE', 'ILMN', 'P',  # added 2026-09-21 (replacing TAP, BLDR, TTD) - verified vs historyofmarket.com / S&P press release
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
    active_sp500_by_year = {y: u for y, u in sp500_by_year.items() if y in RUN_YEARS}
    if not active_sp500_by_year:
        raise SystemExit(f"RUN_YEARS={RUN_YEARS} matches none of the years in sp500_by_year "
                          f"({sorted(sp500_by_year.keys())}). Fix RUN_YEARS and rerun.")
    data_start_date = f'{min(active_sp500_by_year) - 1}-01-01'
    print(f"RUN_YEARS={RUN_YEARS} - downloading data from {data_start_date} onward.")

    all_tickers = sorted(set(
        [t for universe in active_sp500_by_year.values() for t in universe] + [BENCHMARK_TICKER]))
    data = fetch_prices(all_tickers, data_start_date)
    if data.empty:
        raise SystemExit("No price data returned - check network access / tickers. "
                          "(This script needs real yfinance access - it will NOT work "
                          "in a network-sandboxed environment.)")

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
        'sp500_by_year': active_sp500_by_year,
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
