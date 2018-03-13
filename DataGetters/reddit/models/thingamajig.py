from typing import Any, Dict, List, Tuple

HEADLINES_KEYS: Tuple[str, ...] = (
    "domain",
    "subreddit_id",
    "id",
    "likes",
    "view_count",
    "score",
    "downs",
    "title",
    "ups",
    "num_comments",
    "created_utc",
)

COMMENTS_KEYS: Tuple[str, ...] = (
    "created_utc",
    "subreddit_id",
    "id",
    "likes",
    "parent_id",
    "score",
    "downs",
    "body",
    "ups",
    "author",
    "controversiality",
    "depth",
)


class ThingAMaJig (object):
    def __init__(self, keys: Tuple[str, ...]) -> None:
        self._wanted_keys = keys

    def _append_items_from_list(self, items, obj):
        for item in items:
            if isinstance(item, list):
                self._append_items_from_list(item, obj)
            else:
                obj.append(item)

    def _keep_keys(self, item):
        _data = []
        if item["kind"] == "t1" and "replies" in item["data"]:
            if item["data"]["replies"] != "":
                _data.append(self._dict_to_object(item["data"]["replies"]))

        _data.append({k: item["data"][k] if k in item["data"] else None
                      for k in self._wanted_keys})
        return _data

    def _dict_to_object(self, data):
        _data = []
        if "kind" in data:
            start = 1
            if data['data']['children'][0]["kind"] == "t1":
                start = 0
            for x in data['data']['children'][start:]:
                item = self._keep_keys(x)
                if isinstance(item, list):
                    self._append_items_from_list(item, _data)
                else:
                    _data.append(item)
        return _data

    def embody(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        _data: List[Dict[str, Any]] = []
        if isinstance(data, list):
            _data = [self.embody(item) for item in data]
        elif isinstance(data, dict):
            _data = self._dict_to_object(data)
        if len(_data) > 0 and isinstance(_data[0], list):
            return [y for x in _data for y in x if len(x) > 0]
        return _data
