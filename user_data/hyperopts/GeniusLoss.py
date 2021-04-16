from freqtrade.optimize.hyperopt import IHyperOptLoss
import math
from datetime import datetime
from pandas import DataFrame, date_range

# Sortino settings
TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0
MAX_ACCEPTED_TRADE_DURATION = 300
MIN_ACCEPTED_TRADE_DURATION = 2

# Loss settings
EXPECTED_MAX_PROFIT = 3.0
WIN_LOSS_WEIGHT = 5
AVERAGE_PROFIT_WEIGHT = 15
SORTINO_WEIGHT = 0.01
IGNORE_SMALL_PROFITS = False
SMALL_PROFITS_THRESHOLD = 0.001  # 0.1%


def sortino_daily(results: DataFrame, trade_count: int,
                  min_date: datetime, max_date: datetime,
                  *args, **kwargs) -> float:
    """
    Objective function, returns smaller number for more optimal results.

    Uses Sortino Ratio calculation.

    Sortino Ratio calculated as described in
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    resample_freq = '1D'
    slippage_per_trade_ratio = 0.0005
    days_in_year = 365
    minimum_acceptable_return = 0.0

    # apply slippage per trade to profit_ratio
    results.loc[:, 'profit_ratio_after_slippage'] = \
        results['profit_ratio'] - slippage_per_trade_ratio

    # create the index within the min_date and end max_date
    t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                         normalize=True)

    sum_daily = (
        results.resample(resample_freq, on='close_date').agg(
            {"profit_ratio_after_slippage": sum}).reindex(t_index).fillna(0)
    )

    total_profit = sum_daily["profit_ratio_after_slippage"] - minimum_acceptable_return
    expected_returns_mean = total_profit.mean()

    sum_daily['downside_returns'] = 0
    sum_daily.loc[total_profit < 0, 'downside_returns'] = total_profit
    total_downside = sum_daily['downside_returns']
    # Here total_downside contains min(0, P - MAR) values,
    # where P = sum_daily["profit_ratio_after_slippage"]
    down_stdev = math.sqrt((total_downside ** 2).sum() / len(total_downside))

    if down_stdev != 0:
        sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
    else:
        # Define high (negative) sortino ratio to be clear that this is NOT optimal.
        sortino_ratio = -20.

    # print(t_index, sum_daily, total_profit)
    # print(minimum_acceptable_return, expected_returns_mean, down_stdev, sortino_ratio)
    return -sortino_ratio


class GeniusLoss(IHyperOptLoss):
    """
    Defines custom loss function which consider various metrics
    to make more robust strategy.
    Adjust those weights to get more suitable results for your strategy
    WIN_LOSS_WEIGHT
    AVERAGE_PROFIT_WEIGHT
    SORTINO_WEIGHT

    IGNORE_SMALL_PROFITS - this param allow to filter small profits
    (to take into consideration possible spread)
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """
        profit_threshold = 0

        if IGNORE_SMALL_PROFITS:
            profit_threshold = SMALL_PROFITS_THRESHOLD

        total_profit = results['profit_ratio'].sum()
        total_win = len(results[(results['profit_ratio'] > profit_threshold)])
        total_lose = len(results[(results['profit_ratio'] <= 0)])
        average_profit = results['profit_ratio'].mean() * 100
        sortino_ratio = sortino_daily(results, trade_count, min_date, max_date)

        if total_lose == 0:
            total_lose = 1

        profit_loss = 1 - total_profit / EXPECTED_MAX_PROFIT
        win_lose_loss = (1 - (total_win / total_lose)) * WIN_LOSS_WEIGHT
        average_profit_loss = 1 - (average_profit * AVERAGE_PROFIT_WEIGHT)
        sortino_ratio_loss = SORTINO_WEIGHT * sortino_ratio

        result = profit_loss + win_lose_loss + average_profit_loss + sortino_ratio_loss

        return result
