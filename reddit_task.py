import os
import time
from os.path import dirname, join

from dotenv import load_dotenv

from reddit.core import Reddit
from reddit.models.credentials import Credentials
from reddit.models.thingamajig import (
    ThingAMaJig, COMMENTS_KEYS, HEADLINES_KEYS)


if __name__ == '__main__':
    dotenv_path = join(dirname(__file__), 'config/.env')
    load_dotenv(dotenv_path)

    SLEEP = int(os.environ.get("REDDIT_SLEEP"))

    credentials = Credentials(os.environ.get("REDDIT_USERNAME"),
                              os.environ.get("REDDIT_PASSWORD"),
                              os.environ.get("REDDIT_CLIENT_ID"),
                              os.environ.get("REDDIT_CLIENT_SECRET"),
                              os.environ.get("REDDIT_APP_ID"),
                              os.environ.get("REDDIT_APP_VERSION"))
    reddit = Reddit(credentials)

    headlines = ThingAMaJig(HEADLINES_KEYS)
    comments = ThingAMaJig(COMMENTS_KEYS)

    while(True):
        if reddit.login():
            headlines_data = reddit.headlines(os.environ.get(
                "REDDIT_SUBREDDIT"), os.environ.get("REDDIT_LIMIT"))
            headlines_items = headlines.embody(headlines_data["body"])
            headlines_data = None
            comment_items = []
            for item in headlines_items:
                if item["num_comments"] > 0:
                    time.sleep(SLEEP)
                    comment_data = reddit.comments(os.environ.get(
                        "REDDIT_SUBREDDIT"), item["id"], os.environ.get("REDDIT_LIMIT"))
                    comment_items += comments.embody(comment_data["body"])
            comment_items = [x for x in comment_items if len(x) > 0]
        time.sleep(SLEEP)
