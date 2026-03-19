from typing import Union

type JSONValue = Union[
    str, int, float, bool, None, list["JSONValue"], dict[str, "JSONValue"]
]
type JSONObject = dict[str, JSONValue]
