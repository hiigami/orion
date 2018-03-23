import os
import pandas as pd
from pymongo import MongoClient


def delete_item_s_in_collection(collection, items=None):
    if items is None:
        collection.delete_many({})
    elif isinstance(items, []):
        collection.delete_many(items)
    else:
        collection.delete_one(items)


def get_collection(db, name):
    return db[name]


def get_connection():
    MONGO_USER = os.environ.get("MONGO_USER", "")
    MONGO_PWD = os.environ.get("MONGO_PWD", "")
    MONGO_DOM = os.environ.get("MONGO_DOM", "localhost")
    MONGO_PORT = os.environ.get("MONGO_PORT", 27017)
    MONGO_DB = os.environ.get("MONGO_DB", "")

    conn_user_pass = ""
    if MONGO_USER != "":
        conn_user_pass = "{}:{}@".format(MONGO_USER, MONGO_PWD)
    connection_url = "mongodb://{0}{1}:{2}/{3}".format(
        conn_user_pass, MONGO_DOM, MONGO_PORT, MONGO_DB)
    if MONGO_DB == "":
        return MongoClient(connection_url)
    return MongoClient(connection_url)[MONGO_DB]


def get_dataframe(collection, query=None, chunksize=1000, page_num=0, no_id=True):
    '''
    Read from Mongo and Store into a Panda's DataFrame
    '''
    db = get_connection()
    skips = chunksize * page_num
    
    if not query:
        query = {}
    cursor = db[collection].find(query).skip(skips).limit(chunksize)

    df =  pd.DataFrame(list(cursor))

    if no_id:
        del df['_id']

    return df


