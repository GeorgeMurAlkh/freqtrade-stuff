# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import numpy as np
import pandas as pd

buy_params = {
    "informative_fast_length": 20,
    "informative_slow_length": 25,
    "adx_1_multiplier": 1.022,
    "adx_2_multiplier": 1.158,
    "adx_3_multiplier": 1.37,
    "adx_4_multiplier": 1.334,
    "adx_lower_bias": 0.014,
    "adx_middle_bias": 0.033,
    "adx_period": 9,
    "adx_upper_bias": 0.021,
    "downtrend_bias": 0.0,
    "fast_btc_bias": 0.002,
    "ma_lower_length": 27,
    "ma_lower_offset": 0.953,
    "pair_is_bad_1_threshold": 0.12,
    "pair_is_bad_2_threshold": 0.09,
    "slow_btc_bias": 0.007,
}

sell_params = {
    "ma_middle_1_length": 30,
    "ma_middle_1_offset": 0.985,
    "ma_upper_length": 16,
    "ma_upper_offset": 1.014,
}


class MADisplaceV4(IStrategy):

    ma_lower_length = IntParameter(15, 30, default=buy_params['ma_lower_length'], space='buy')
    ma_lower_offset = DecimalParameter(0.94, 0.98, default=buy_params['ma_lower_offset'], space='buy')

    informative_fast_length = IntParameter(15, 35, default=buy_params['informative_fast_length'], space='disable')
    informative_slow_length = IntParameter(20, 40, default=buy_params['informative_slow_length'], space='disable')

    adx_period = CategoricalParameter([2, 3, 4, 5, 7, 9, 10, 12, 14, 20], default=buy_params['adx_period'], space='buy')
    adx_1_multiplier = DecimalParameter(1, 1.2, default=buy_params['adx_1_multiplier'], space='buy')
    adx_2_multiplier = DecimalParameter(1.1, 1.4, default=buy_params['adx_2_multiplier'], space='buy')
    adx_3_multiplier = DecimalParameter(1.2, 1.6, default=buy_params['adx_3_multiplier'], space='buy')
    adx_4_multiplier = DecimalParameter(1.3, 1.9, default=buy_params['adx_4_multiplier'], space='buy')

    adx_lower_bias = DecimalParameter(0.005, 0.035, default=buy_params['adx_lower_bias'], space='buy')
    adx_middle_bias = DecimalParameter(0.005, 0.035, default=buy_params['adx_middle_bias'], space='buy')
    adx_upper_bias = DecimalParameter(0.005, 0.05, default=buy_params['adx_upper_bias'], space='buy')

    slow_btc_bias = DecimalParameter(0, 0.03, default=buy_params['slow_btc_bias'], space='buy')
    fast_btc_bias = DecimalParameter(0, 0.03, default=buy_params['fast_btc_bias'], space='buy')
    downtrend_bias = DecimalParameter(0, 0.03, default=buy_params['downtrend_bias'], space='buy')

    pair_is_bad_1_threshold = DecimalParameter(0, 0.12, default=buy_params['pair_is_bad_1_threshold'], space='buy')
    pair_is_bad_2_threshold = DecimalParameter(0, 0.09, default=buy_params['pair_is_bad_2_threshold'], space='buy')

    ma_middle_1_length = IntParameter(15, 35, default=sell_params['ma_middle_1_length'], space='sell')
    ma_middle_1_offset = DecimalParameter(0.93, 1.005, default=sell_params['ma_middle_1_offset'], space='sell')
    ma_upper_length = IntParameter(15, 25, default=sell_params['ma_upper_length'], space='sell')
    ma_upper_offset = DecimalParameter(1.005, 1.025, default=sell_params['ma_upper_offset'], space='sell')

    minimal_roi = {"0": 1}

    stoploss = -0.2

    # you can use this protections with lower stoploss (like stoploss = -0.1) instead of
    # (dataframe['pair_is_bad'] == 0) in buy conditions
    # protections = [
    #     {
    #         "method": "StoplossGuard",
    #         "lookback_period": 60,
    #         "trade_limit": 1,
    #         "stop_duration": 60,
    #         "only_per_pair": True
    #     },
    #     {
    #         "method": "StoplossGuard",
    #         "lookback_period": 60,
    #         "trade_limit": 2,
    #         "stop_duration": 60,
    #         "only_per_pair": False
    #     }
    # ]

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True

    plot_config = {
        'main_plot': {
            'ma_lower': {'color': 'red'},
            'ma_middle_1': {'color': 'green'},
            'ma_upper': {'color': 'pink'},
        },
        'subplots': {
            'Trend': {
                'pair_is_bad': {'color: red'}
            }
        }
    }

    use_custom_stoploss = True
    startup_candle_count = 200

    informative_timeframe = '1h'

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        if self.config['runmode'].value == 'hyperopt':
            dataframe = self.informative_dataframe.copy()
            dataframe_btc = self.informative_dataframe_btc_1h
            dataframe_5m_btc = self.informative_dataframe_btc_5m
        else:
            dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
            dataframe_btc = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.informative_timeframe)
            dataframe_5m_btc = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.timeframe)

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=int(self.informative_fast_length.value))
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=int(self.informative_slow_length.value))

        dataframe['ema_fast_btc'] = ta.EMA(dataframe_btc, timeperiod=int(self.informative_fast_length.value))
        dataframe['ema_slow_btc'] = ta.EMA(dataframe_btc, timeperiod=int(self.informative_slow_length.value))

        dataframe['ema_fast_5m_btc'] = ta.EMA(dataframe_5m_btc, timeperiod=50)
        dataframe['ema_slow_5m_btc'] = ta.EMA(dataframe_5m_btc, timeperiod=200)
        dataframe['ema_delta_5m_btc'] = (
                    (dataframe['ema_fast_5m_btc'] - dataframe['ema_slow_5m_btc']) / dataframe['ema_fast_5m_btc'])

        dataframe['downtrend'] = (
            (dataframe['ema_fast'] < dataframe['ema_slow'])
        ).astype('int')

        dataframe['downtrend_btc'] = (
            (dataframe['ema_fast_btc'] < dataframe['ema_slow_btc'])
        ).astype('int')

        dataframe['downtrend_fast_btc'] = (
                (dataframe['ema_fast_5m_btc'] < dataframe['ema_slow_5m_btc'])
                &
                (dataframe['ema_delta_5m_btc'] < -0.0025)
        ).astype('int')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < -0.04 and current_time - timedelta(minutes=35) > trade.open_date_utc:
            return -0.01

        return -0.99

    def get_main_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=int(self.adx_period.value))
        adx_conditions = [
            (dataframe['adx'] < 25),
            (dataframe['adx'] >= 25) & (dataframe['adx'] < 50),
            (dataframe['adx'] >= 50) & (dataframe['adx'] < 75),
            (dataframe['adx'] >= 75)
        ]
        adx_choices = [
            self.adx_1_multiplier.value,
            self.adx_2_multiplier.value,
            self.adx_3_multiplier.value,
            self.adx_4_multiplier.value
        ]
        dataframe['adx_multiplier'] = np.select(adx_conditions, adx_choices, default=1)

        slow_btc_bias = (dataframe['downtrend_btc'] * self.slow_btc_bias.value)
        fast_btc_bias = (dataframe['downtrend_fast_btc'] * self.fast_btc_bias.value)
        downtrend_bias = (dataframe['downtrend'] * self.downtrend_bias.value)

        total_downtrend_bias = slow_btc_bias + fast_btc_bias + downtrend_bias

        adx_lower_bias = self.adx_lower_bias.value
        adx_middle_bias = self.adx_middle_bias.value
        adx_upper_bias = self.adx_upper_bias.value

        lower_offset_downtrend = (
                ((-total_downtrend_bias + -adx_lower_bias) * dataframe['adx_multiplier']) + (
                    self.ma_lower_offset.value + adx_lower_bias + total_downtrend_bias))
        middle_1_offset_downtrend = (
                ((-total_downtrend_bias + -adx_middle_bias) * dataframe['adx_multiplier']) + (
                    self.ma_middle_1_offset.value + adx_middle_bias + total_downtrend_bias))
        upper_offset_downtrend = (
                ((-total_downtrend_bias + -adx_upper_bias) * dataframe['adx_multiplier']) + (
                    self.ma_upper_offset.value + adx_upper_bias + total_downtrend_bias))

        dataframe['ma_lower'] = ta.SMA(dataframe,
                                       timeperiod=int(self.ma_lower_length.value)) * lower_offset_downtrend
        dataframe['ma_middle_1'] = ta.SMA(dataframe,
                                          timeperiod=int(self.ma_middle_1_length.value)) * middle_1_offset_downtrend
        dataframe['ma_upper'] = ta.SMA(dataframe,
                                           timeperiod=int(self.ma_upper_length.value)) * upper_offset_downtrend

        dataframe['pair_is_bad'] = ((((dataframe['open'].shift(12) - dataframe['close']) / dataframe[
            'close']) >= self.pair_is_bad_1_threshold.value) |
                                    (((dataframe['open'].shift(6) - dataframe['close']) / dataframe[
                                        'close']) >= self.pair_is_bad_2_threshold.value)).astype('int')

        # drop NAN in hyperopt to fix "'<' not supported between instances of 'str' and 'int' error
        if self.config['runmode'].value == 'hyperopt':
            dataframe = dataframe.dropna()

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            self.informative_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'],
                                                                    timeframe=self.informative_timeframe)

            self.informative_dataframe_btc_1h = self.dp.get_pair_dataframe(pair='BTC/USDT',
                                                                           timeframe=self.informative_timeframe)

            self.informative_dataframe_btc_5m = self.dp.get_pair_dataframe(pair='BTC/USDT', timeframe=self.timeframe)

        if self.config['runmode'].value != 'hyperopt':
            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # it calling multiple times and dataframe overrides same columns
        # so check if calculated column already existing
        if self.config['runmode'].value == 'hyperopt' and 'downtrend' not in dataframe:
            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)
            pd.options.mode.chained_assignment = None

        dataframe.loc[
            (
                (dataframe['pair_is_bad'] == 0)
                &
                (dataframe['close'] < dataframe['ma_lower'])
                &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value == 'hyperopt' and 'downtrend' not in dataframe:
            informative = self.get_informative_indicators(metadata)
            dataframe = self.merge_informative(informative, dataframe)
            dataframe = self.get_main_indicators(dataframe, metadata)
            pd.options.mode.chained_assignment = None

        dataframe.loc[
            (
                (
                    (dataframe['close'] > dataframe['ma_upper'])
                    |
                    (qtpylib.crossed_below(dataframe['close'], dataframe['ma_middle_1']))

                )
                &
                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe

    def merge_informative(self, informative: DataFrame, dataframe: DataFrame) -> DataFrame:

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)

        # don't overwrite the base dataframe's HLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        return dataframe

# Recommended config
# "max_open_trades": 3,
# "pair_blacklist": [
#             "BNB/BTC",
#             "BUSD/USDT",
#             "EPS/USDT",
#             "BNB*/.*",
#             ".*BEAR/USDT",
#             ".*BULL/USDT",
#             ".*UP/USDT",
#             ".*DOWN/USDT",
#             ".*HEDGE/USDT",
#             "USDC/USDT",
#             "EUR/USDT",
#             "TUSD/USDT"
#         ]
# "pairlists": [
#         {
#             "method": "VolumePairList",
#             "number_assets": 80,
#             "sort_key": "quoteVolume",
#             "refresh_period": 1440
#         },
#         {"method": "AgeFilter", "min_days_listed": 10},
#         {"method": "PrecisionFilter"},
#         {"method": "SpreadFilter", "max_spread_ratio": 0.005},
#         {
#             "method": "RangeStabilityFilter",
#             "lookback_days": 10,
#             "min_rate_of_change": 0.01,
#             "refresh_period": 1440
#         }
#     ]