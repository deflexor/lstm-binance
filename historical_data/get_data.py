from binance import Client
import numpy as np

intervals = {240:'4h', 60:'1h', 15:'15m', 5:'5m', 1: '1m'}
start = '14 days ago UTC'
trading_pair = 'TRXUSDT'

# load key and secret and connect to API
keys = open('./keys.txt').readline().split(' ')
print('Connecting to Client...')
api = Client(keys[0], keys[1])

# fetch desired candles of all data
print('Fetching data (may take multiple API requests)')
hist = api.get_historical_klines(trading_pair, intervals[1], start)
print('Finished.')

# create numpy object with closing prices and volume
hist = np.array(hist, dtype=np.float32)
hist = hist[:, 4]

# data information
print("\nDatapoints:  {0}".format(hist.shape[0]))
print("Memory:      {0:.2f} Mb\n".format((hist.nbytes) / 1000000))

# save to file as numpy object
print(hist)
np.save("hist_data", hist)
