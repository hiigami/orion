import platform
import time
from datetime import datetime, timedelta

from utils.requests import Request


class Reddit(object):

    def __init__(self, credentials):
        self._token = None
        self._token_url = "https://www.reddit.com/api/v1/access_token"
        self._credentials = credentials

    def _user_agent(self):
        return "{0}:{1}:{2} (by /u/{3})".format(platform.system(),
                                                self._credentials.app_id,
                                                self._credentials.version,
                                                self._credentials.username)

    def _header(self, include_token=True):
        headers = {
            "User-Agent": self._user_agent()
        }
        if include_token:
            headers["Authorization"] = "{0} {1}".format(self._token['token_type'],
                                                        self._token["access_token"])
        return headers

    def _auth(self):
        return (self._credentials.client_id, self._credentials.client_secret)

    def _request(self, url, data=None, method="GET"):
        return Request.run(url,
                           method,
                           data,
                           self._header(),
                           self._auth())

    def _is_logged(self):
        if self._token is not None and self._token['expires_date'] < datetime.now():
            return True
        return False

    def login(self):
        if self._is_logged():
            return True

        data = {
            "grant_type": "password",
            "username": self._credentials.username,
            "password": self._credentials.password
        }
        response = Request.run(self._token_url, "POST",
                               data, self._header(False), self._auth())
        if response['statusCode'] == 200:
            self._token = response['body']
            self._token['expires_date'] = datetime.fromtimestamp(time.mktime(
                datetime.now().timetuple())) + timedelta(seconds=int(self._token['expires_in']))
            return True
        return False

    def headlines(self, subreddit, limit, sort="new"):
        url = "https://www.reddit.com/r/{0}/{1}/.json?limit={2}" \
            .format(subreddit,
                    sort,
                    limit)
        return self._request(url)

    def comments(self, subreddit, comment_id, limit):
        url = "https://www.reddit.com/r/{0}/comments/{1}/.json?limit={2}" \
            .format(subreddit,
                    comment_id,
                    limit)
        return self._request(url)
