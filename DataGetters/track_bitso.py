import time
import json
import os
from .bitso import api as api
import utils.mongo as mongo
from utils.load_env import load
from utils.logger import getLogger 

logger = getLogger(__name__)

def save(collection_name: str, data: dict) -> None:
    db = mongo.get_connection()
    collection = mongo.get_collection(
            db, os.environ.get(collection_name))
    logger.info("Saving a %s with id %d", collection_name, data['_id'])
    collection.insert_one(data)

def save_ticker(t: int) -> None:
    data = api.get_ticker()
    data['_id'] = t
    del data['volume']
    del data['vwap']
    del data['high']
    del data['book']
    del data['low']
    save('tickers', data)

def save_order_book(t: int) -> None:
    data = api.get_order_book()
    data['_id'] = t
    del data['sequence']
    for el in data['asks']:
        del el['book']
        if 'oid' in el:
            del el['oid']
    for el in data['bids']:
        del el['book']
        if 'oid' in el:
            del el['oid']
    save('order_books', data)

def save_trades(t: int) -> None:
    data = {'_id': t}
    data['trades'] = api.get_trades()
    for el in data['trades']:
        del el['book']
        del el['tid']
    save('trades', data)

def save_bitso_records() -> None:
    t = int(time.time())
    save_ticker(t)
    save_order_book(t)
    save_trades(t)

def main_bitso(*args, **kwargs) -> None:
    load()
    SLEEP = int(os.environ.get("BITSO_SLEEP", 40))
    _continue = True
    while(_continue):
        save_bitso_records()
        time.sleep(SLEEP)
