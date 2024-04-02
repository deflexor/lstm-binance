import asyncio
from collections import deque
import json
from binance import AsyncClient, DepthCacheManager, BinanceSocketManager
from binance.helpers import round_step_size
from binance.enums import *
import numpy as np
import keras
from lstm import extract_data, shape_data

tick_sizes = {
   'TRXUSDT': 10,
   'DOGEUSDT': 0.0001,
}

TIMESTEPS = 10

trading_pair = 'TRXUSDT'
# no newline in file!
keys = open('./keys.txt').readline().split(' ')

ORDER = None

def run_prediction(data):
  c = list(map(lambda d: float(d['c']), data))
  X, y = extract_data(np.array(c))
  X, y = shape_data(X, y, timesteps=TIMESTEPS)

  model = keras.saving.load_model('models/lstm_model.keras')
  # RUN model
  prediction = model(X)
  return(prediction)

def create_order(client, msg, side = SIDE_BUY):
  amount = 1
  tick_size = tick_sizes[trading_pair]
  rounded_amount = round_step_size(amount, tick_size)
  print("creating order:")
  print(f"ramt: {rounded_amount}, side:{side}, price:{msg.c}")
  order = client.create_margin_order(
    symbol=trading_pair,
    side=side,
    type=ORDER_TYPE_LIMIT,
    timeInForce=TIME_IN_FORCE_GTC,
    quantity=rounded_amount,
    price=msg.c)
  print("create order:")
  print(order)

def manage_order(client, predict, msg):
  print(f"prediction result0: {predict[:,0]}")
  print(f"prediction result1: {predict[:,1]}")
  global ORDER
  if ORDER is not None:
    close_test_order(client, msg)
  else:
    if prediction > 0.5:
      create_test_order(client, msg, SIDE_BUY)
    else:
      create_test_order(client, msg, SIDE_SELL)

def create_test_order(client, msg, side = SIDE_BUY):
  global ORDER
  sl_k = 0.005
  sl = msg.c + msg.c * sl_k if side == SIDE_SELL else msg.c - msg.c * sl_k
  ORDER = {
    "side": SIDE_BUY,
    "price": msg.c,
    "sl": sl,
    "status": ORDER_STATUS_FILLED,
  }

def close_test_order(client, msg):
  global ORDER
  global TOT_PROFIT
  profit = ORDER['price'] - msg.c if ORDER['side'] == SIDE_SELL else msg.c - ORDER['price']
  slprofit = ORDER['sl'] - msg.c if ORDER['side'] == SIDE_SELL else msg.c - ORDER['sl']
  sl_k = 0.005
  if profit > ORDER['price'] * 0.015: sl_k = 0.01
  if profit > ORDER['price'] * 0.05: sl_k = 0.025
  if profit > ORDER['price'] * 0.1: sl_k = 0.002
  sl = msg.c + msg.c * sl_k if ORDER["side"] == SIDE_SELL else msg.c - msg.c * sl_k
  print(f"profit: {profit}, sl: {sl}, slprofit: {slprofit}")
  if slprofit < 0:
    # close order!
    TOT_PROFIT.push(profit)
    ORDER = None
  else:
    # move sl
    ORDER["sl"] = sl



async def kline_listener(bsm, ipcqueue, client):
  async with bsm.kline_socket(symbol=trading_pair, interval=AsyncClient.KLINE_INTERVAL_1MINUTE) as stream:
    while True:
      res = await stream.recv()
      #print(f'kline_socket recv {res}')
      await ipcqueue.put(res['k'])

async def trade_listener(bsm, ipcqueue, client):
  async with bsm.margin_socket() as stream:
    while True:
      res = await stream.recv()
      print(f'trade_listener margin_socket recv {res}')

async def queue_listener(ipcqueue, client):
  tencandles = deque([], 60)
  while True:
    # Get a "work item" out of the queue.
    item = await ipcqueue.get()
    #print(f'got from queue: {item} {len(tencandles)}')
    if len(tencandles) == 0 or item['t'] > tencandles[len(tencandles)-1]['t']:
      tencandles.append(item)
      if len(tencandles) == tencandles.maxlen:
        print(f"prediction run!")
        predict = run_prediction(tencandles)
        # TODO: put buy order or sell order depending on result
        manage_order(client, predict[:,0], tencandles[len(tencandles)-1])

    # Notify the queue that the "work item" has been processed.
    ipcqueue.task_done()



async def main():
  client = await AsyncClient.create(keys[0], keys[1])
  #print(json.dumps(await client.get_symbol_ticker(symbol=trading_pair), indent=2))
  queue = asyncio.Queue()
  bsm = BinanceSocketManager(client)
  # orders = await client.get_open_margin_orders(symbol=trading_pair)
  # aorders = await client.get_all_margin_orders(symbol=trading_pair)
  # print("my orders:")
  # print(orders)
  # print("my all orders:")
  # print(aorders)

  async for kline in await client.get_historical_klines_generator(trading_pair, AsyncClient.KLINE_INTERVAL_1MINUTE, "60 minutes ago UTC"):
    await queue.put({ "t": kline[0], "o": kline[1], "h": kline[2], "l": kline[3], "c": kline[4], "v": kline[5], "T": kline[6]})

  await asyncio.gather(
     kline_listener(bsm, queue, client),
     trade_listener(bsm, queue, client),
     queue_listener(queue, client),
  )

  await client.close_connection()




if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
