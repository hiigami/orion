import json
from typing import Any, Callable, Dict, Tuple

import requests


class Method(object):
    GET = requests.get
    POST = requests.post

    @staticmethod
    def validate(item) -> bool:
        if item == Method.GET or item == Method.POST:
            return True
        return False


class Request(object):

    @staticmethod
    def method(method):
        if isinstance(method, str):
            return getattr(Method, method)
        elif Method.validate(method):
            return method
        raise ValueError("Not a valid str or request method")

    @classmethod
    def run(cls,
            url: str,
            method,
            data: Dict[str, Any] = None,
            headers: Dict[str, str] = None,
            auth: Tuple[str, str] = None,
            time_out: int = None) -> Dict[str, Any]:
        request = cls.method(method)
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
        except requests.exceptions.HTTPError as error:
            return {'statusCode': req.status_code, 'body': error.response}
