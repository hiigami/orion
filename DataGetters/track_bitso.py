import time
import json
from bitso import public_api as api

def save(filename, data):
  # ToDo - The files are temporary placed here. They are
  #        going to be stored in a MongoDB later.
  f = open(filename, 'a')
  f.write(json.dumps(data) + ',\n')
  f.close()

def save_ticker(t):
  data = api.get_ticker()
  data['date'] = t
  # ToDo - Just save last, ask and bid properties
  save('tickers.txt', data)

def save_order_book(t):
  data = api.get_order_book()
  data['date'] = t
  save('order_books.txt', data)

def save_trades(t):
  data = {'date': t}
  data['trades'] = api.get_trades()
  save('trades.txt', data)


def save_bitso_records():
  t = time.time()
  save_ticker(t)
  save_order_book(t)
  save_trades(t)

if __name__ == '__main__':
  while(True):
    save_bitso_records()
    time.sleep(10)
