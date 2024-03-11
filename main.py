import asyncio
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

trading_pair = 'TRXUSDT'
# no newline in file!
keys = open('./keys.txt').readline().split(' ')

def on_data(data):
  X, y = extract_data(np.array(data['close']))
  X, y = shape_data(X, y, timesteps=10)

  model = keras.saving.load_model('models/lstm_model.keras')
  # RUN model
  prediction = model(X)
  print(prediction)

def create_order(client, msg, side = SIDE_BUY):
  amount = 1
  tick_size = tick_sizes[trading_pair]
  rounded_amount = round_step_size(amount, tick_size)
  print("creating order:")
  print(f"amt: {}", rounded_amount)
  order = client.create_margin_order(
    symbol=trading_pair,
    side=side,
    type=ORDER_TYPE_LIMIT,
    timeInForce=TIME_IN_FORCE_GTC,
    quantity=rounded_amount,
    price=msg.c)
  print("create order:")
  print(order)

async def kline_listener(bsm, client):
  async with bsm.kline_socket(symbol=trading_pair, interval=AsyncClient.KLINE_INTERVAL_1MINUTE) as stream:
    #while True:
    for _ in range(5):
      res = await stream.recv()
      print(f'kline_socket recv {res}')

async def trade_listener(bsm, client):
  async with bsm.margin_socket() as stream:
    while True:
      res = await stream.recv()
      print(f'margin_socket recv {res}')

async def main():
  client = await AsyncClient.create(keys[0], keys[1])
  #print(json.dumps(await client.get_symbol_ticker(symbol=trading_pair), indent=2))
  queue = asyncio.Queue()
  bsm = BinanceSocketManager(client)


  orders = await client.get_open_margin_orders(symbol=trading_pair)
  aorders = await client.get_all_margin_orders(symbol=trading_pair)
  print("my orders:")
  print(orders)
  print("my all orders:")
  print(aorders)

  async for kline in await client.get_historical_klines_generator(trading_pair, AsyncClient.KLINE_INTERVAL_1MINUTE, "9 minutes ago UTC"):
        print(kline)


  await asyncio.gather(
     kline_listener(bsm, client),
     trade_listener(bsm, client),
  )
  #task = asyncio.create_task(some_coro(param=i))

  await client.close_connection()




if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
