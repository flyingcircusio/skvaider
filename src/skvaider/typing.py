from collections.abc import Sequence

type JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | Sequence["JSONValue"]
    | dict[str, "JSONValue"]
)
type JSONObject = dict[str, JSONValue]


type ConfigValue = JSONValue
type ConfigDict = JSONObject
