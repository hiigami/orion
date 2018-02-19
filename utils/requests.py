import json

import requests


class Method(object):
    GET = requests.get
    POST = requests.post


class Request(object):

    @staticmethod
    def _method(method):
        return getattr(Method, method)

    @classmethod
    def run(cls,
            url,
            method,
            data=None,
            headers={},
            auth=None,
            time_out=None):
        request = cls._method(method)
        try:
            req = request(url,
                          data=data,
                          auth=auth,
                          headers=headers,
                          timeout=time_out)
            response = ""
            try:
                response = req.json()
            except:
                response = req.text
            return {'statusCode': req.status_code, 'body': response}
        except requests.exceptions.Timeout:
            raise Exception("Gateway Timeout")
        except requests.exceptions.HTTPError as a:
            return {'statusCode': req.status_code, 'body': a.message}
