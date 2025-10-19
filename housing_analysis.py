#!/usr/bin/env python3
"""
Housing Market Analysis
-----------------------
Reads Zillow-style time series CSVs (State, Metro, County optional) and produces:
- YoY price trends (ZHVI_AllHomes)
- Demand proxies (Inventory, Days on Zillow) and a Demand Index
- National median series (median across states per month)
- CSV exports; optional charts (PNG)

Usage:
    python housing_analysis.py \
        --state /path/State_time_series.csv \
        --metro /path/Metro_time_series.csv \
        --county /path/County_time_series.csv \
        --dict /path/DataDictionary.csv \
        --outdir ./outputs \
        --save-charts

Notes:
- Only --state is strictly required. Others are optional.
- If --save-charts is set, figures are saved as PNG in outdir/figs.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe for servers/CI
import matplotlib.pyplot as plt


PRICE_COL = "ZHVI_AllHomes"
INVENTORY_SA = "InventorySeasonallyAdjusted_AllHomes"
INVENTORY_RAW = "InventoryRaw_AllHomes"
DAYS_COL = "DaysOnZillow_AllHomes"
REGION_COL = "RegionName"
DATE_COL = "Date"


def read_csv_safely(path: Path, parse_dates=None):
    """Read CSV with robust encoding fallbacks."""
    encodings = [None, "utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, parse_dates=parse_dates)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Could not read {path}: {last_exc}")


def yoy_pct(series: pd.Series) -> pd.Series:
    """Year-over-year percentage change for monthly data."""
    return series.pct_change(12) * 100.0


def ensure_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set Date as monthly start frequency index (no interpolation)."""
    if DATE_COL not in df.columns:
        raise ValueError("Expected a 'Date' column in the data.")
    out = df.set_index(DATE_COL).sort_index()
    # We won't reindex to avoid filling; just set freq for resampling-aware ops
    out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp("M")
    return out


def compute_region_metrics(g: pd.DataFrame) -> pd.Series:
    """For a single region's time-series, compute latest YoY metrics and latest levels."""
    ts = ensure_monthly_index(g)
    out = {}

    # Price YoY and latest
    if PRICE_COL in ts:
        out["YoY_Price_%"] = yoy_pct(ts[PRICE_COL]).iloc[-1] if len(ts) >= 13 else np.nan
        out["LatestPrice"] = ts[PRICE_COL].iloc[-1]

    # Inventory (prefer SA over raw)
    inv_col = INVENTORY_SA if INVENTORY_SA in ts.columns else (INVENTORY_RAW if INVENTORY_RAW in ts.columns else None)
    if inv_col:
        out["YoY_Inventory_%"] = yoy_pct(ts[inv_col]).iloc[-1] if len(ts) >= 13 else np.nan
        out["LatestInventory"] = ts[inv_col].iloc[-1]

    # Days on Zillow
    if DAYS_COL in ts:
        out["YoY_Days_%"] = yoy_pct(ts[DAYS_COL]).iloc[-1] if len(ts) >= 13 else np.nan
        out["LatestDays"] = ts[DAYS_COL].iloc[-1]

    return pd.Series(out)


def build_trends(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    """Compute per-region latest trends for a given geography level (state/metro/county)."""
    needed = [DATE_COL, REGION_COL, PRICE_COL]
    if not all(c in df.columns for c in needed):
        missing = [c for c in needed if c not in df.columns]
        raise ValueError(f"{level_name} file missing columns: {missing}")

    data = df.dropna(subset=[DATE_COL]).copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL])
    data = data.sort_values([REGION_COL, DATE_COL])

    trends = (
        data.groupby(REGION_COL, group_keys=False)
            .apply(compute_region_metrics)
            .reset_index()
    )

    # Demand Index: combine standardized YoY inventory and YoY days (declines -> stronger demand)
    for col in ["YoY_Inventory_%", "YoY_Days_%"]:
        if col in trends.columns:
            z = (trends[col] - trends[col].mean(skipna=True)) / trends[col].std(skipna=True)
            trends[col + "_z"] = z

    if "YoY_Inventory_%_z" in trends.columns and "YoY_Days_%_z" in trends.columns:
        trends["DemandIndex"] = (-trends["YoY_Inventory_%_z"] + -trends["YoY_Days_%_z"]) / 2.0

    trends["GeoLevel"] = level_name
    return trends


def national_series_from_states(states: pd.DataFrame) -> pd.DataFrame:
    """Construct a national median ZHVI by taking the cross-sectional median of state values each month."""
    s = states[[DATE_COL, REGION_COL, PRICE_COL]].dropna(subset=[DATE_COL]).copy()
    s[DATE_COL] = pd.to_datetime(s[DATE_COL], errors="coerce")
    s = s.dropna(subset=[DATE_COL])
    piv = s.pivot_table(index=DATE_COL, columns=REGION_COL, values=PRICE_COL, aggfunc="first")
    piv = piv.sort_index()
    nat = pd.DataFrame({"ZHVI_National_Median": piv.median(axis=1, skipna=True)})
    nat["YoY_%"] = yoy_pct(nat["ZHVI_National_Median"])
    nat.index.name = DATE_COL
    return nat.reset_index()


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_charts(outdir: Path, national: pd.DataFrame, state_trends: pd.DataFrame):
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # 1) National trend
    if national is not None and not national.empty:
        plt.figure()
        d = pd.to_datetime(national[DATE_COL])
        plt.plot(d, national["ZHVI_National_Median"])
        plt.title("National Median Home Value (ZHVI)")
        plt.xlabel("Date")
        plt.ylabel("USD")
        save_fig(figs / "national_median_zhvi.png")

    # 2) Top 10 states by YoY ZHVI
    if state_trends is not None and not state_trends.empty and "YoY_Price_%" in state_trends.columns:
        top10 = state_trends.nlargest(10, "YoY_Price_%")[["RegionName", "YoY_Price_%"]].sort_values("YoY_Price_%")
        plt.figure()
        plt.barh(top10["RegionName"], top10["YoY_Price_%"])
        plt.title("Top 10 States by YoY ZHVI Growth (Latest)")
        plt.xlabel("YoY %")
        plt.ylabel("State")
        save_fig(figs / "top10_states_yoy_price.png")

    # 3) Scatter: Inventory YoY vs Price YoY
    if state_trends is not None and not state_trends.empty and "YoY_Inventory_%" in state_trends.columns:
        st = state_trends.dropna(subset=["YoY_Inventory_%", "YoY_Price_%"])
        if not st.empty:
            plt.figure()
            plt.scatter(st["YoY_Inventory_%"], st["YoY_Price_%"])
            plt.title("States: Inventory YoY vs Price YoY")
            plt.xlabel("Inventory YoY %")
            plt.ylabel("Price YoY %")
            save_fig(figs / "states_inventory_vs_price.png")

    # 4) Demand Index top 10
    if state_trends is not None and "DemandIndex" in state_trends.columns:
        top_demand = state_trends.nlargest(10, "DemandIndex")[["RegionName", "DemandIndex"]].sort_values("DemandIndex")
        plt.figure()
        plt.barh(top_demand["RegionName"], top_demand["DemandIndex"])
        plt.title("Top 10 States by Demand Index (Lower Inventory & Faster DOM)")
        plt.xlabel("Demand Index (higher = stronger)")
        plt.ylabel("State")
        save_fig(figs / "top10_states_demand_index.png")


def main():
    ap = argparse.ArgumentParser(description="Housing Market Analysis (Zillow-style time series).")
    ap.add_argument("--state", type=Path, required=True, help="Path to State_time_series.csv")
    ap.add_argument("--metro", type=Path, default=None, help="Path to Metro_time_series.csv (optional)")
    ap.add_argument("--county", type=Path, default=None, help="Path to County_time_series.csv (optional)")
    ap.add_argument("--dict", type=Path, default=None, help="Path to DataDictionary.csv (optional)")
    ap.add_argument("--outdir", type=Path, default=Path("./outputs"), help="Output directory for CSVs and (optional) charts")
    ap.add_argument("--save-charts", action="store_true", help="Save PNG charts to outdir/figs")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load core files
    print(f"Reading state data: {args.state}")
    state = read_csv_safely(args.state, parse_dates=[DATE_COL])

    metro = None
    if args.metro and args.metro.exists():
        print(f"Reading metro data: {args.metro}")
        metro = read_csv_safely(args.metro, parse_dates=[DATE_COL])

    county = None
    if args.county and args.county.exists():
        print(f"Reading county data: {args.county}")
        county = read_csv_safely(args.county, parse_dates=[DATE_COL])

    # Dictionary is optional; load if provided (for your own exploration)
    data_dictionary = None
    if args.dict and args.dict.exists():
        try:
            print(f"Reading data dictionary: {args.dict}")
            data_dictionary = read_csv_safely(args.dict)
        except Exception as e:
            print(f"Skipping dictionary due to read error: {e}")

    # ---- Build trends ----
    print("Computing state-level trends...")
    state_trends = build_trends(state, level_name="state")
    state_trends_out = args.outdir / "state_trends_summary.csv"
    state_trends.to_csv(state_trends_out, index=False)
    print(f"Saved: {state_trends_out}")

    metro_trends = None
    if metro is not None:
        try:
            print("Computing metro-level trends...")
            metro_trends = build_trends(metro, level_name="metro")
            metro_trends_out = args.outdir / "metro_trends_summary.csv"
            metro_trends.to_csv(metro_trends_out, index=False)
            print(f"Saved: {metro_trends_out}")
        except Exception as e:
            print(f"Metro trends skipped: {e}")

    # ---- National series ----
    print("Computing national median series from states...")
    national = national_series_from_states(state)
    national_out = args.outdir / "national_median_zhvi.csv"
    national.to_csv(national_out, index=False)
    print(f"Saved: {national_out}")

    # ---- Charts ----
    if args.save_charts:
        print("Saving charts...")
        make_charts(args.outdir, national, state_trends)
        print(f"Charts saved in: {args.outdir / 'figs'}")

    # ---- Console summary ----
    def safe_top(df, k, col, desc):
        try:
            top = df.nlargest(k, col)[[REGION_COL, col]]
            print(f"\nTop {k} by {desc}:")
            print(top.to_string(index=False))
        except Exception:
            pass

    safe_top(state_trends, 10, "YoY_Price_%", "YoY Price Growth (States)")
    if "DemandIndex" in state_trends.columns:
        safe_top(state_trends, 10, "DemandIndex", "Demand Index (States)")

    if metro_trends is not None:
        safe_top(metro_trends, 10, "YoY_Price_%", "YoY Price Growth (Metros)")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
