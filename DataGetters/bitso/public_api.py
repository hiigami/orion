import requests
import json

BOOK = 'btc_mxn'
API = 'https://api.bitso.com/v3'
SERVICES = {
  'get_ticker': '/ticker/',
  'order_book': '/order_book/',
  'trades': '/trades/'
}

def get_ticker():
  URL = API + SERVICES['get_ticker']
  res = requests.get(URL, {'book': BOOK}).text
  return json.loads(res)['payload']

def get_order_book(): 
  URL = API + SERVICES['order_book']
  res = requests.get(URL, {'book': BOOK}).text
  return json.loads(res)['payload']

def get_trades():
  URL = API + SERVICES['trades']
  res = requests.get(URL, {'book': BOOK, 'limit': 100}).text
  return json.loads(res)['payload']
