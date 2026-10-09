"""Free-data factor research pilot.

Uses BaoStock's free API to build a deterministic 60-stock sample from the
current CSI 300, constructs simplified CNE5-style risk exposures, neutralizes
a volume-confirmed reversal alpha, and evaluates IC plus a five-quantile
five-day rebalance backtest.

This is a workflow demonstration, not an official MSCI Barra reproduction.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "work" / "pydeps"))

import baostock as bs
import numpy as np
import pandas as pd


START_DATE = "2022-01-01"
END_DATE = "2026-09-30"
UNIVERSE_DATE = "2026-09-30"
TOP_INDUSTRIES = 10
N_PER_INDUSTRY = 6
SEED = 42
MARKET_CODE = "sh.000300"
ONE_WAY_COST = 0.0015

WORK_DIR = ROOT / "work" / "free_factor_pilot"
PRICE_CACHE = WORK_DIR / "prices"
SHARE_CACHE = WORK_DIR / "shares"
OUT_DIR = ROOT / "outputs" / "free_factor_pilot"

PRICE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,isST"
)


def ensure_dirs() -> None:
    for path in (WORK_DIR, PRICE_CACHE, SHARE_CACHE, OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def login() -> None:
    response = bs.login()
    if response.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {response.error_msg}")


def result_to_frame(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    return pd.DataFrame(rows, columns=result.fields)


def build_universe() -> pd.DataFrame:
    path = WORK_DIR / "universe.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"code": str})

    hs300 = result_to_frame(bs.query_hs300_stocks(UNIVERSE_DATE))
    industries = result_to_frame(bs.query_stock_industry(date=UNIVERSE_DATE))
    universe = hs300.merge(
        industries[["code", "industry", "industryClassification"]],
        on="code", how="left",
    )
    universe = universe[universe["industry"].fillna("").ne("")].copy()
    leading = universe["industry"].value_counts().head(TOP_INDUSTRIES).index
    universe = universe[universe["industry"].isin(leading)]
    sampled = (
        universe.groupby("industry", group_keys=False)
        .sample(n=N_PER_INDUSTRY, random_state=SEED)
        .reset_index(drop=True)
    )
    sampled = sampled.sort_values(["industry", "code"]).reset_index(drop=True)
    sampled.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"universe: {len(sampled)} stocks across {sampled['industry'].nunique()} industries", flush=True)
    return sampled


def fetch_prices(code: str) -> pd.DataFrame:
    path = PRICE_CACHE / f"{code.replace('.', '_')}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    last_error = None
    for attempt in range(3):
        result = bs.query_history_k_data_plus(
            code, PRICE_FIELDS, start_date=START_DATE, end_date=END_DATE,
            frequency="d", adjustflag="3",
        )
        if result.error_code == "0":
            frame = result_to_frame(result)
            if not frame.empty:
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                frame["date"] = pd.to_datetime(frame["date"])
                return frame
        last_error = result.error_msg
        time.sleep(1 + attempt)
    raise RuntimeError(f"price fetch failed for {code}: {last_error}")


def fetch_annual_shares(code: str) -> pd.DataFrame:
    path = SHARE_CACHE / f"{code.replace('.', '_')}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["pubDate", "statDate"], dtype={"code": str})
    frames = []
    for year in range(2021, 2026):
        result = bs.query_profit_data(code=code, year=year, quarter=4)
        if result.error_code == "0":
            frame = result_to_frame(result)
            if not frame.empty:
                frames.append(frame)
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data = data[["code", "pubDate", "statDate", "totalShare", "liqaShare"]]
        data["pubDate"] = pd.to_datetime(data["pubDate"], errors="coerce")
        data["statDate"] = pd.to_datetime(data["statDate"], errors="coerce")
        data["totalShare"] = pd.to_numeric(data["totalShare"], errors="coerce")
        data["liqaShare"] = pd.to_numeric(data["liqaShare"], errors="coerce")
        data = data.dropna(subset=["pubDate", "totalShare"]).sort_values("pubDate")
    else:
        data = pd.DataFrame(columns=["code", "pubDate", "statDate", "totalShare", "liqaShare"])
    data.to_csv(path, index=False, encoding="utf-8-sig")
    return data


def download_data(universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = []
    shares = []
    codes = universe["code"].tolist()
    for index, code in enumerate(codes, 1):
        prices.append(fetch_prices(code))
        shares.append(fetch_annual_shares(code))
        if index % 10 == 0 or index == len(codes):
            print(f"downloaded {index}/{len(codes)} stocks", flush=True)
    market = fetch_prices(MARKET_CODE)
    price_data = pd.concat(prices, ignore_index=True)
    share_frames = [frame for frame in shares if not frame.empty]
    share_data = pd.concat(share_frames, ignore_index=True) if share_frames else pd.DataFrame()
    return price_data, share_data, market


def numeric_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    numeric = [
        "open", "high", "low", "close", "preclose", "volume", "amount",
        "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "isST",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def exp_weights(length: int, half_life: float) -> np.ndarray:
    age = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / half_life)
    return weights / weights.sum()


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * values))
    return float(np.sqrt(np.sum(weights * np.square(values - mean))))


def beta_hsigma(stock: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    weights = exp_weights(len(stock), 63)
    xm = float(np.sum(weights * market))
    ym = float(np.sum(weights * stock))
    xc = market - xm
    denominator = float(np.sum(weights * xc * xc))
    if denominator <= 0:
        return np.nan, np.nan
    beta = float(np.sum(weights * xc * (stock - ym)) / denominator)
    alpha = ym - beta * xm
    residual = stock - alpha - beta * market
    return beta, weighted_std(residual, weights)


def cmra(returns: np.ndarray) -> float:
    if len(returns) != 252:
        return np.nan
    logs = np.log1p(np.clip(returns, -0.999999, None))
    monthly = [logs[-21 * (i + 1): len(logs) - 21 * i if i else None].sum() for i in range(12)]
    cumulative = np.cumsum(monthly)
    return float(cumulative.max() - cumulative.min())


def attach_point_in_time_shares(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for code, group in prices.groupby("code"):
        group = group.sort_values("date")
        history = shares[shares["code"] == code].sort_values("pubDate") if not shares.empty else pd.DataFrame()
        if history.empty:
            group["totalShare"] = np.nan
        else:
            group = pd.merge_asof(
                group, history[["pubDate", "totalShare"]],
                left_on="date", right_on="pubDate", direction="backward",
            ).drop(columns=["pubDate"])
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def calculate_raw_descriptors(
    prices: pd.DataFrame, shares: pd.DataFrame, market: pd.DataFrame,
) -> pd.DataFrame:
    prices = numeric_prices(prices)
    market = numeric_prices(market).sort_values("date")
    market["market_ret"] = market["pctChg"] / 100.0
    market_ret = market.set_index("date")["market_ret"]
    prices = attach_point_in_time_shares(prices, shares)

    outputs = []
    for code, group in prices.groupby("code"):
        group = group.sort_values("date").copy()
        group["ret"] = group["pctChg"] / 100.0
        group["market_ret"] = group["date"].map(market_ret)
        group["turn_frac"] = group["turn"] / 100.0
        group["log_mcap_raw"] = np.log(group["close"] * group["totalShare"])

        group["stom_raw"] = np.log(group["turn_frac"].rolling(21, min_periods=21).sum())
        group["stoq_raw"] = np.log(group["turn_frac"].rolling(63, min_periods=63).sum() / 3.0)
        group["stoa_raw"] = np.log(group["turn_frac"].rolling(252, min_periods=252).sum() / 12.0)
        group["earnings_yield_raw"] = np.where(group["peTTM"] > 0, 1.0 / group["peTTM"], np.nan)
        group["book_to_price_raw"] = np.where(group["pbMRQ"] > 0, 1.0 / group["pbMRQ"], np.nan)

        log_ret = np.log1p(np.clip(group["ret"], -0.999999, None))
        ret5 = np.expm1(log_ret.rolling(5, min_periods=5).sum())
        turn5 = group["turn_frac"].rolling(5, min_periods=5).mean()
        turn60 = group["turn_frac"].rolling(60, min_periods=60).mean()
        turnover_shock = np.log(turn5 / turn60).clip(lower=0)
        group["alpha_raw"] = -ret5 * turnover_shock

        for horizon in (5, 20):
            shifted = log_ret.shift(-1)
            future_logs = shifted.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
            group[f"fwd_{horizon}d"] = np.expm1(future_logs)

        beta_values = np.full(len(group), np.nan)
        hsigma_values = np.full(len(group), np.nan)
        dastd_values = np.full(len(group), np.nan)
        cmra_values = np.full(len(group), np.nan)
        momentum_values = np.full(len(group), np.nan)
        stock_returns = group["ret"].to_numpy(float)
        market_returns = group["market_ret"].to_numpy(float)
        for position in range(len(group)):
            if position >= 251:
                stock_window = stock_returns[position - 251: position + 1]
                market_window = market_returns[position - 251: position + 1]
                if np.isfinite(stock_window).all() and np.isfinite(market_window).all():
                    beta_values[position], hsigma_values[position] = beta_hsigma(stock_window, market_window)
                    dastd_values[position] = weighted_std(stock_window, exp_weights(252, 42))
                    cmra_values[position] = cmra(stock_window)
            if position >= 524:
                window = stock_returns[position - 524: position - 20]
                if len(window) == 504 and np.isfinite(window).all():
                    momentum_values[position] = float(
                        np.sum(exp_weights(504, 126) * np.log1p(np.clip(window, -0.999999, None)))
                    )
        group["beta_raw"] = beta_values
        group["hsigma_raw"] = hsigma_values
        group["dastd_raw"] = dastd_values
        group["cmra_raw"] = cmra_values
        group["momentum_raw"] = momentum_values
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def winsor_z(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=series.index)
    lower, upper = valid.quantile([0.01, 0.99])
    clipped = series.clip(lower, upper)
    scale = clipped.std(ddof=0)
    return (clipped - clipped.mean()) / scale if scale > 0 else pd.Series(0.0, index=series.index)


def daily_z(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", group_keys=False)[column].apply(winsor_z)


def orthogonalize(frame: pd.DataFrame, target: str, controls: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("date"):
        valid = group[[target] + controls].dropna()
        if len(valid) <= len(controls) + 2:
            continue
        design = np.column_stack([np.ones(len(valid))] + [valid[c].to_numpy(float) for c in controls])
        values = valid[target].to_numpy(float)
        residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        scale = residuals.std(ddof=0)
        if scale > 0:
            result.loc[valid.index] = (residuals - residuals.mean()) / scale
    return result


def build_exposures(raw: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    data = raw.merge(universe[["code", "code_name", "industry"]], on="code", how="inner")
    source_columns = [
        "log_mcap_raw", "beta_raw", "momentum_raw", "dastd_raw", "cmra_raw",
        "hsigma_raw", "stom_raw", "stoq_raw", "stoa_raw",
        "earnings_yield_raw", "book_to_price_raw", "alpha_raw",
    ]
    for column in source_columns:
        data[column.removesuffix("_raw") + "_z"] = daily_z(data, column)

    data["resvol_pre_z"] = (
        0.74 * data["dastd_z"] + 0.16 * data["cmra_z"] + 0.10 * data["hsigma_z"]
    )
    data["resvol_z"] = orthogonalize(data, "resvol_pre_z", ["beta_z"])
    data["liquidity_z"] = 0.35 * data["stom_z"] + 0.35 * data["stoq_z"] + 0.30 * data["stoa_z"]
    data["nlsize_cube"] = data["log_mcap_z"] ** 3
    data["nlsize_z"] = orthogonalize(data, "nlsize_cube", ["log_mcap_z"])

    controls = [
        "log_mcap_z", "nlsize_z", "beta_z", "momentum_z", "resvol_z",
        "liquidity_z", "earnings_yield_z", "book_to_price_z",
    ]
    data["alpha_neutral_z"] = neutralize_alpha(data, controls)
    return data


def neutralize_alpha(data: pd.DataFrame, controls: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby("date"):
        industries = pd.get_dummies(group["industry"], prefix="ind", drop_first=True, dtype=float)
        work = pd.concat([group[["alpha_z"] + controls], industries], axis=1).dropna()
        if len(work) <= len(controls) + industries.shape[1] + 3:
            continue
        x_columns = controls + industries.columns.tolist()
        design = np.column_stack([np.ones(len(work)), work[x_columns].to_numpy(float)])
        values = work["alpha_z"].to_numpy(float)
        residuals = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        scale = residuals.std(ddof=0)
        if scale > 0:
            result.loc[work.index] = (residuals - residuals.mean()) / scale
    return result


def rank_ic(group: pd.DataFrame, factor: str, target: str) -> float:
    valid = group[[factor, target]].dropna()
    if len(valid) < 20:
        return np.nan
    return float(valid[factor].rank().corr(valid[target].rank()))


def evaluate_ic(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, group in data.groupby("date"):
        row = {"date": date}
        for factor in ("alpha_z", "alpha_neutral_z"):
            for horizon in (5, 20):
                row[f"{factor}_ic_{horizon}d"] = rank_ic(group, factor, f"fwd_{horizon}d")
        rows.append(row)
    series = pd.DataFrame(rows).sort_values("date")
    summary = {}
    for column in series.columns.drop("date"):
        values = series[column].dropna()
        std = values.std(ddof=1)
        summary[column] = {
            "observations": int(len(values)),
            "mean_ic": float(values.mean()) if len(values) else None,
            "std_ic": float(std) if len(values) else None,
            "icir": float(values.mean() / std) if len(values) and std > 0 else None,
            "positive_ratio": float((values > 0).mean()) if len(values) else None,
        }
    return series, summary


def quantile_backtest(data: pd.DataFrame) -> pd.DataFrame:
    usable = data.dropna(subset=["alpha_neutral_z", "fwd_5d"]).copy()
    dates = sorted(usable["date"].unique())[::5]
    previous_weights: dict[int, dict[str, float]] = {group: {} for group in range(1, 6)}
    equity = {group: 1.0 for group in range(1, 6)}
    rows = []
    for date in dates:
        cross = usable[usable["date"] == date].copy()
        if len(cross) < 25:
            continue
        cross["quantile"] = pd.qcut(
            cross["alpha_neutral_z"].rank(method="first"), 5, labels=False,
        ).astype(int) + 1
        row = {"date": date}
        for group_id in range(1, 6):
            members = cross[cross["quantile"] == group_id]
            weights = {code: 1.0 / len(members) for code in members["code"]}
            old = previous_weights[group_id]
            all_codes = set(old) | set(weights)
            turnover = (
                1.0 if not old
                else 0.5 * sum(abs(weights.get(code, 0) - old.get(code, 0)) for code in all_codes)
            )
            gross = float(members["fwd_5d"].mean())
            net = gross - ONE_WAY_COST * turnover
            equity[group_id] *= 1 + net
            row[f"q{group_id}_gross_return"] = gross
            row[f"q{group_id}_turnover"] = turnover
            row[f"q{group_id}_net_return"] = net
            row[f"q{group_id}_equity"] = equity[group_id]
            previous_weights[group_id] = weights
        row["long_short_net_return"] = (
            row["q5_gross_return"] - row["q1_gross_return"]
            - ONE_WAY_COST * (row["q5_turnover"] + row["q1_turnover"])
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["long_short_equity"] = (1 + result["long_short_net_return"]).cumprod()
    return result


def write_report(
    universe: pd.DataFrame, exposures: pd.DataFrame, ic_summary: dict,
    groups: pd.DataFrame,
) -> None:
    first_date = exposures.dropna(subset=["alpha_neutral_z"])["date"].min().date()
    last_date = exposures.dropna(subset=["alpha_neutral_z"])["date"].max().date()
    neutral5 = ic_summary["alpha_neutral_z_ic_5d"]
    neutral20 = ic_summary["alpha_neutral_z_ic_20d"]
    raw5 = ic_summary["alpha_z_ic_5d"]
    raw20 = ic_summary["alpha_z_ic_20d"]
    q_final = {f"Q{i}": float(groups[f"q{i}_equity"].iloc[-1]) for i in range(1, 6)}
    ls_final = float(groups["long_short_equity"].iloc[-1])
    values = list(q_final.values())
    monotonic = all(a <= b for a, b in zip(values, values[1:])) or all(
        a >= b for a, b in zip(values, values[1:])
    )
    verdict = (
        "候选Alpha未通过本次小样本有效性检验：中性化后的平均IC接近0，"
        "五组最终净值不呈单调关系，Q5-Q1研究多空组合最终净值低于1。"
        if abs(neutral5["mean_ic"]) < 0.03 and not monotonic and ls_final < 1
        else "候选Alpha显示出部分统计信号，但仍需扩大样本并进行样本外验证。"
    )
    text = f"""# 免费接口小样本因子研究报告

## 研究设置

- 股票池：2026-09-30 沪深300成分股中，样本数量最多的10个证监会行业分类，每个行业固定随机抽取6只，共{len(universe)}只。
- 原始数据：BaoStock免费接口，2022-01-01至2026-09-30。
- 有效因子检验期：{first_date}至{last_date}。
- 候选Alpha：放量下跌反转，即 `-过去5日收益 × max(ln(5日平均换手率 / 60日平均换手率), 0)`。
- 风险控制：行业、Size、Non-linear Size、Beta、Momentum、Residual Volatility、Liquidity、盈利收益率代理和账面市值比代理。
- 未来收益：5日和20日；信号使用当日及以前数据，未来收益从下一交易日开始。
- 分组回测：5组、每5个交易日调仓；单边交易成本假设为{ONE_WAY_COST:.2%}。

## IC结果

| 指标 | 原始5日 | 原始20日 | 中性化5日 | 中性化20日 |
|---|---:|---:|---:|---:|
| 平均Rank IC | {raw5['mean_ic']:.4f} | {raw20['mean_ic']:.4f} | {neutral5['mean_ic']:.4f} | {neutral20['mean_ic']:.4f} |
| ICIR | {raw5['icir']:.4f} | {raw20['icir']:.4f} | {neutral5['icir']:.4f} | {neutral20['icir']:.4f} |
| IC为正比例 | {raw5['positive_ratio']:.2%} | {raw20['positive_ratio']:.2%} | {neutral5['positive_ratio']:.2%} | {neutral20['positive_ratio']:.2%} |

## 五组回测最终净值

| 分组 | 最终净值 |
|---|---:|
"""
    for label, value in q_final.items():
        text += f"| {label} | {value:.4f} |\n"
    text += f"| Q5-Q1研究多空组合 | {ls_final:.4f} |\n"
    text += f"""

## 本次结论

{verdict}

这意味着程序链路已经得到验证，但不能因为流程成功就把该信号当成有效Alpha。下一步应检查因子定义、持有周期和适用市场状态，再提出新的经济假设；不应通过反复挑参数把本次结果修饰成正收益。

## 行业因子的处理

本次已经把10个证监会行业类别转换成0/1行业暴露，并放入每日横截面回归，用于剔除候选Alpha中的行业影响。行业暴露另存为 `行业因子暴露.csv`。

这只是行业中性化控制，并非正式CNE5行业模型：CNE5使用32个GICS行业因子，并需要每天通过股票收益横截面回归估计行业因子收益。本样本只有10个行业、60只股票，也没有完整的历史GICS分类，因此没有声称复现了Barra行业因子收益。

## 解释边界

- 这是免费数据小样本的流程验证，不是MSCI官方Barra模型。
- 股票池使用2026年9月的成分和行业分类，存在幸存者偏差与历史成分偏差。
- Size使用最近已公布年度财报中的总股本与当日不复权价格构造，两个年报之间的股本变化可能未及时反映。
- 盈利收益率和账面市值比分别用正PE、正PB的倒数近似，不含分析师预测。
- 行业采用2026-09-30的证监会行业分类，没有完整复原每一天的历史行业变更。
- 分组多空结果用于研究比较；A股个股做空受到现实约束。
- 当前交易成本是简化假设，尚未建立个股冲击成本模型。
- 任何正向结果都需要在历史动态股票池、其他时间区间和更大样本上做样本外验证。
"""
    (OUT_DIR / "研究报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    login()
    try:
        universe = build_universe()
        prices, shares, market = download_data(universe)
    finally:
        bs.logout()

    print("calculating descriptors", flush=True)
    raw = calculate_raw_descriptors(prices, shares, market)
    exposures = build_exposures(raw, universe)
    ic_series, ic_summary = evaluate_ic(exposures)
    groups = quantile_backtest(exposures)

    universe.to_csv(OUT_DIR / "样本股票池.csv", index=False, encoding="utf-8-sig")
    industry_exposure = pd.concat(
        [
            universe[["code", "code_name", "industry", "industryClassification"]],
            pd.get_dummies(universe["industry"], prefix="industry", dtype=int),
        ],
        axis=1,
    )
    industry_exposure.to_csv(OUT_DIR / "行业因子暴露.csv", index=False, encoding="utf-8-sig")
    exposure_columns = [
        "date", "code", "code_name", "industry", "close", "alpha_raw", "alpha_z",
        "alpha_neutral_z", "fwd_5d", "fwd_20d", "log_mcap_z", "nlsize_z",
        "beta_z", "momentum_z", "resvol_z", "liquidity_z",
        "earnings_yield_z", "book_to_price_z",
    ]
    exposures[exposure_columns].to_csv(
        OUT_DIR / "因子暴露与未来收益.csv", index=False, encoding="utf-8-sig",
    )
    ic_series.to_csv(OUT_DIR / "IC序列.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "IC汇总.json").write_text(
        json.dumps(ic_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    groups.to_csv(OUT_DIR / "五组回测.csv", index=False, encoding="utf-8-sig")
    write_report(universe, exposures, ic_summary, groups)
    print(f"completed: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
