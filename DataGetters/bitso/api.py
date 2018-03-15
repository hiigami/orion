import requests
import json
from typing import Dict, Any

BOOK = 'btc_mxn'
API = 'https://api.bitso.com/v3'
SERVICES = {
    'get_ticker': '/ticker/',
    'order_book': '/order_book/',
    'trades': '/trades/'
}

def get_btc_data(service: str, 
            params: Dict[str, Any] = None) -> Dict[str, Any]:
    if not params:
        params = {'book': BOOK, 'limit': 100}
    URL = API + SERVICES[service]
    res = requests.get(URL, params).text
    return json.loads(res)['payload']

def get_ticker() -> Dict[str, Any]:
    return get_btc_data('get_ticker')


def get_order_book() -> Dict[str, Any]:
    return get_btc_data('order_book')


def get_trades() -> Dict[str, Any]:
    return get_btc_data('trades')
