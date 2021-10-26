from pathlib import Path
from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import ExchangeResolver
from freqtrade.plugins.pairlistmanager import PairListManager
import pandas as pd
from datetime import datetime, timedelta
import argparse
from dateutil.relativedelta import *
import json
import os

STAKE_CURRENCY = 'BUSD'

config = Configuration.from_files([])
config["dataformat_ohlcv"] = "hdf5"
config["timeframe"] = "1d"
config['exchange']['name'] = "binance"
config['stake_currency'] = STAKE_CURRENCY
config['exchange']['pair_whitelist'] = [
    f'.*/{STAKE_CURRENCY}',
]
config['exchange']['pair_blacklist'] = [
    '^(.*USD|USDC|AUD|BRZ|CAD|CHF|EUR|GBP|HKD|SGD|TRY|ZAR|TUSD)/.*',
    'PAX/.*',
    'DAI/.*',
    'PAXG/.*',
    ".*UP/USDT",
    ".*DOWN/USDT",
    ".*BEAR/USDT",
    ".*BULL/USDT"
]
config['pairlists'] = [
    {
        "method": "StaticPairList",
    },
]

exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
pairlists = PairListManager(exchange, config)
pairlists.refresh_pairlist()
pairs = pairlists.whitelist
data_location = Path(config['user_data_dir'], 'data', config['exchange']['name'])

print(f"found {str(len(pairs))} pairs on {config['exchange']['name']}")

DATE_FORMAT = '%Y%m%d'
DATE_TIME_FORMAT = '%Y%m%d %H:%M:%S'

def get_data_slices_dates(df, start_date_str, end_date_str, interval):
    # df_start_date = df.date.min()
    # df_end_date = df.date.max()

    defined_start_date = datetime.strptime(start_date_str, DATE_TIME_FORMAT)
    defined_end_date = datetime.strptime(end_date_str, DATE_TIME_FORMAT)

    # start_date = df_start_date if defined_start_date < df_start_date else defined_start_date
    # end_date = df_end_date if defined_end_date > df_end_date else defined_end_date

    start_date = defined_start_date
    end_date = defined_end_date

    # time_delta = timedelta(hours=interval_hr)
    if interval == 'monthly':
        time_delta = relativedelta(months=+1)
    elif interval == 'weekly':
        time_delta = relativedelta(weeks=+1)
    elif interval == 'daily':
        time_delta = relativedelta(days=+1)
    else:
        time_delta = relativedelta(months=+1)

    slices = []

    run = True

    while run:
        # slice_start_time = end_date - time_delta
        slice_end_time = start_date + time_delta
        if slice_end_time <= end_date:
            slice_date = {
                'start': start_date,
                'end': slice_end_time
            }

            slices.append(slice_date)
            start_date = slice_end_time
        else:
            slice_date = {
                'start': start_date,
                'end': defined_end_date
            }

            slices.append(slice_date)
            run = False

    return slices


def process_candles_data(pairs, filter_price):
    full_dataframe = pd.DataFrame()

    for pair in pairs:


        print(data_location)
        print(config["timeframe"])
        print(pair)

        candles = load_pair_history(
            datadir=data_location,
            timeframe=config["timeframe"],
            pair=pair,
            data_format="hdf5"
        )

        if len(candles):
            # Not sure about AgeFilter
            # apply price filter make price 0 to ignore this pair after calculation of quoteVolume
            candles.loc[(candles.close < filter_price), 'close'] = 0
            column_name = pair
            candles[column_name] = candles['volume'] * candles['close']

            if full_dataframe.empty:
                full_dataframe = candles[['date', column_name]].copy()
            else:
                full_dataframe = pd.merge(full_dataframe, candles[['date', column_name]].copy(), on='date', how='left')
            # print("Loaded " + str(len(candles)) + f" rows of data for {pair} from {data_location}")
            # print(full_dataframe.tail(1))

    print(full_dataframe.head())

    full_dataframe['date'] = full_dataframe['date'].dt.tz_localize(None)

    return full_dataframe


def process_date_slices(df, date_slices, number_assets):
    result = {}
    for date_slice in date_slices:
        df_slice = df[(df.date >= date_slice['start']) & (df.date < date_slice['end'])].copy()
        summarised = df_slice.sum()
        summarised = summarised[summarised > 0]
        summarised = summarised.sort_values(ascending=False)

        if len(summarised) > number_assets:
            result_pairs_list = list(summarised.index[:number_assets])
        else:
            result_pairs_list = list(summarised.index)

        if len(result_pairs_list) > 0:
            result[f'{date_slice["start"].strftime(DATE_FORMAT)}-{date_slice["end"].strftime(DATE_FORMAT)}'] = result_pairs_list

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="config to parse")
    parser.add_argument("-t", "--timerange", nargs='?',
                        help="timerange as per freqtrade format, e.g. 20210401-, 20210101-20210201, etc")
    parser.add_argument("-o", "--outfile", help="path where output the pairlist", type=argparse.FileType('w'))
    parser.add_argument("-mp", "--minprice", help="price for price filter")
    parser.add_argument("-tf", "--timeframe", help="timeframe of loaded candles data")
    parser.add_argument("-na", "--numberassets", help="number of assets to be filtered")
    args = parser.parse_args()

    # Make this argparseble
    # And add blacklist
    START_DATE_STR = '20180101 00:00:00'
    END_DATE_STR = '20211001 00:00:00'
    # For now it shouldn't be less than a day because it's outputs object with timerage suitable for backtesting
    # for easy copying eg. 20210501-20210602
    INTERVAL_ARR = ['monthly', 'weekly', 'daily']
    # INTERVAL_ARR = ['weekly']
    # INTERVAL_ARR = ['monthly']
    ASSET_FILTER_PRICE_ARR = [0, 0.01, 0.02, 0.05, 0.15, 0.5]
    NUMBER_ASSETS_ARR = [30, 45, 60, 75, 90, 105, 120]

    # ASSET_FILTER_PRICE_ARR = [0]
    # NUMBER_ASSETS_ARR = [90]

    start_string = START_DATE_STR.split(' ')[0]
    end_string = END_DATE_STR.split(' ')[0]


    for asset_filter_price in ASSET_FILTER_PRICE_ARR:

        volume_dataframe = process_candles_data(pairs, asset_filter_price)

        for interval in INTERVAL_ARR:
            date_slices = get_data_slices_dates(volume_dataframe, START_DATE_STR, END_DATE_STR, interval)

            for number_assets in NUMBER_ASSETS_ARR:

                result_obj = process_date_slices(volume_dataframe, date_slices, number_assets)
                # {'timerange': [pairlist]}
                print(result_obj)
                p_json = json.dumps(result_obj, indent=4)
                file_name = f'user_data/pairlists/{STAKE_CURRENCY}/{interval}/{interval}_{number_assets}_{STAKE_CURRENCY}_{str(asset_filter_price).replace(".", ",")}_minprice_{start_string}_{end_string}.json'
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, 'w') as f:
                    f.write(p_json)


    # Save result object as json to --outfile location
    print('Done.')

if __name__ == "__main__":
    main()

