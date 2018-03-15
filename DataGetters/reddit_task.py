import hashlib
import json
from utils.logger import getLogger 
import os
import time
from datetime import date, datetime
from typing import List

import utils.mongo as mongo
from utils.load_env import load

from .reddit.core import Reddit
from .reddit.models.credentials import Credentials
from .reddit.models.thingamajig import (COMMENTS_KEYS, HEADLINES_KEYS,
                                        ThingAMaJig)

logger = getLogger(__name__)


def _get_hash(item) -> str:
    tmp = json.dumps(item).encode('utf-8')
    return hashlib.md5(tmp).hexdigest()


def _update_record(old, new):
    old_hash = None
    new_hash = None
    if "posts" in old:
        old_hash = _get_hash(old["posts"])
    if "posts" in new:
        new_hash = _get_hash(new["posts"])
    if old_hash is None and new_hash is not None:
        return new
    if old_hash == new_hash:
        return None
    if "posts" not in new:
        new["posts"] = old["posts"]
        return new
    for x in old["posts"]:
        is_missing = True
        for y in new["posts"]:
            if x["id"] == y["id"]:
                is_missing = False
                break
        if is_missing:
            new["posts"].append(x)
    return new


def _save_comments(items) -> None:
    db = mongo.get_connection()
    collection = mongo.get_collection(
        db, os.environ.get("REDDIT_DB_COLLECTION"))

    for item in items:
        item["post_id"] = item.pop("id")
        record = collection.find_one({
            "post_id": item["post_id"],
            "subreddit_id": item["subreddit_id"]
        })
        if record:
            item = _update_record(record, item)
            if item is not None:
                collection.replace_one(record, item)
                logger.info("Updated reddit comment with id: %s",
                            record["_id"])
        else:
            comment_id = collection.insert_one(item).inserted_id
            logger.info("Save reddit comment with id: %s", comment_id)


def _reddit_conn() -> Reddit:
    credentials = Credentials(os.environ.get("REDDIT_USERNAME"),
                              os.environ.get("REDDIT_PASSWORD"),
                              os.environ.get("REDDIT_CLIENT_ID"),
                              os.environ.get("REDDIT_CLIENT_SECRET"),
                              os.environ.get("REDDIT_APP_ID"),
                              os.environ.get("REDDIT_APP_VERSION"))
    return Reddit(credentials)


def _main_loop(reddit: Reddit,
               headlines: ThingAMaJig,
               comments: ThingAMaJig,
               SLEEP: int) -> None:
    _continue = True
    while(_continue):
        if reddit.login():
            headlines_data = reddit.headlines(os.environ.get("REDDIT_SUBREDDIT"),
                                              int(os.environ.get("REDDIT_LIMIT")))
            headlines_items = headlines.embody(headlines_data["body"])
            headlines_data = None
            for item in headlines_items:
                if item["num_comments"] > 0:
                    time.sleep(SLEEP)
                    comment_data = reddit.comments(os.environ.get("REDDIT_SUBREDDIT"),
                                                   item["id"],
                                                   int(os.environ.get("REDDIT_LIMIT")))
                    item["posts"] = comments.embody(comment_data["body"])
            _save_comments(headlines_items)
        #_continue = False
        time.sleep(SLEEP)


def main(*args, **kwargs) -> None:
    load()
    SLEEP = int(os.environ.get("REDDIT_SLEEP", 5))
    reddit = _reddit_conn()
    headlines = ThingAMaJig(HEADLINES_KEYS)
    comments = ThingAMaJig(COMMENTS_KEYS)
    _main_loop(reddit, headlines, comments, SLEEP)
