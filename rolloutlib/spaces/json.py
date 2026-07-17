"""Single-value JSON codecs and JSON Schema generation for Gymnasium spaces."""

from __future__ import annotations

from functools import singledispatch
from typing import Any, cast

import numpy as np
from gymnasium import Space
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Text,
    Tuple,
)

from ._pydantic import PydanticSpace


JSONValue = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONSchema = dict[str, Any]


def _require_member(space: Space[Any], value: object) -> None:
    if value not in space:
        raise ValueError(f"value is outside {space!r}")


@singledispatch
def to_json_value(space: Space[Any], value: Any) -> JSONValue:
    """Encode one value from a Gymnasium space as JSON-compatible Python data.

    Custom spaces can register an implementation with
    ``to_json_value.register(CustomSpace)``.
    """

    _require_member(space, value)
    encoded = space.to_jsonable([value])
    if not isinstance(encoded, list) or len(encoded) != 1:
        raise NotImplementedError(
            f"{type(space).__name__} does not expose a single-value JSON encoding; "
            "register a to_json_value implementation"
        )
    return cast(JSONValue, encoded[0])


@to_json_value.register
def _(space: Discrete, value: Any) -> JSONValue:
    _require_member(space, value)
    return int(value)


@to_json_value.register
def _(space: Box, value: Any) -> JSONValue:
    _require_member(space, value)
    return cast(JSONValue, np.asarray(value, dtype=space.dtype).tolist())


@to_json_value.register
def _(space: MultiBinary, value: Any) -> JSONValue:
    _require_member(space, value)
    return cast(JSONValue, np.asarray(value, dtype=space.dtype).tolist())


@to_json_value.register
def _(space: MultiDiscrete, value: Any) -> JSONValue:
    _require_member(space, value)
    return cast(JSONValue, np.asarray(value, dtype=space.dtype).tolist())


@to_json_value.register
def _(space: Text, value: Any) -> JSONValue:
    _require_member(space, value)
    return cast(str, value)


@to_json_value.register
def _(space: Dict, value: Any) -> JSONValue:
    _require_member(space, value)
    return {
        key: to_json_value(child, value[key])
        for key, child in space.spaces.items()
    }


@to_json_value.register
def _(space: Tuple, value: Any) -> JSONValue:
    _require_member(space, value)
    return [
        to_json_value(child, item)
        for child, item in zip(space.spaces, value, strict=True)
    ]


@to_json_value.register
def _(space: Sequence, value: Any) -> JSONValue:
    _require_member(space, value)
    return [to_json_value(space.feature_space, item) for item in value]


@to_json_value.register
def _(space: PydanticSpace, value: Any) -> JSONValue:
    _require_member(space, value)
    return cast(JSONValue, space.adapter.dump_python(value, mode="json"))


@singledispatch
def from_json_value(space: Space[Any], value: JSONValue) -> Any:
    """Decode and validate one JSON-compatible value for a Gymnasium space.

    Custom spaces can register an implementation with
    ``from_json_value.register(CustomSpace)``.
    """

    decoded = space.from_jsonable([value])
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise NotImplementedError(
            f"{type(space).__name__} does not expose a single-value JSON decoding; "
            "register a from_json_value implementation"
        )
    result = decoded[0]
    _require_member(space, result)
    return result


@from_json_value.register
def _(space: Discrete, value: JSONValue) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("expected an integer")
    decoded = space.dtype.type(value)
    _require_member(space, decoded)
    return decoded


@from_json_value.register
def _(space: Box, value: JSONValue) -> Any:
    array = np.asarray(value, dtype=space.dtype)
    _require_member(space, array)
    return array


@from_json_value.register
def _(space: MultiBinary, value: JSONValue) -> Any:
    array = np.asarray(value, dtype=space.dtype)
    _require_member(space, array)
    return array


@from_json_value.register
def _(space: MultiDiscrete, value: JSONValue) -> Any:
    array = np.asarray(value, dtype=space.dtype)
    _require_member(space, array)
    return array


@from_json_value.register
def _(space: Text, value: JSONValue) -> Any:
    if not isinstance(value, str):
        raise ValueError("expected a string")
    _require_member(space, value)
    return value


@from_json_value.register
def _(space: Dict, value: JSONValue) -> Any:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    if set(value) != set(space.spaces):
        raise ValueError("object keys do not match the Dict space")
    decoded = {
        key: from_json_value(child, value[key])
        for key, child in space.spaces.items()
    }
    _require_member(space, decoded)
    return decoded


@from_json_value.register
def _(space: Tuple, value: JSONValue) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != len(space.spaces):
        raise ValueError("expected an array matching the Tuple space")
    decoded = tuple(
        from_json_value(child, item)
        for child, item in zip(space.spaces, value, strict=True)
    )
    _require_member(space, decoded)
    return decoded


@from_json_value.register
def _(space: Sequence, value: JSONValue) -> Any:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected an array")
    decoded = tuple(from_json_value(space.feature_space, item) for item in value)
    _require_member(space, decoded)
    return decoded


@from_json_value.register
def _(space: PydanticSpace, value: JSONValue) -> Any:
    decoded = space.validate(value)
    _require_member(space, decoded)
    return decoded


def _array_schema(
    scalar_schema: JSONSchema,
    shape: tuple[int, ...],
) -> JSONSchema:
    schema = scalar_schema
    for length in reversed(shape):
        schema = {
            "type": "array",
            "items": schema,
            "minItems": length,
            "maxItems": length,
        }
    return schema


@singledispatch
def to_json_schema(space: Space[Any]) -> JSONSchema:
    """Return a JSON Schema describing values accepted by a Gymnasium space.

    The generated schema is intended for model tool arguments and other JSON
    interfaces. Unsupported custom spaces fail explicitly and can register a
    specialized implementation with ``to_json_schema.register(CustomSpace)``.
    """

    raise NotImplementedError(
        f"JSON Schema generation is not implemented for {type(space).__name__}"
    )


@to_json_schema.register
def _(space: Discrete) -> JSONSchema:
    return {
        "type": "integer",
        "minimum": int(space.start),
        "maximum": int(space.start + space.n - 1),
    }


@to_json_schema.register
def _(space: Box) -> JSONSchema:
    scalar_type = "integer" if np.issubdtype(space.dtype, np.integer) else "number"
    scalar: JSONSchema = {"type": scalar_type}
    if np.all(np.isfinite(space.low)) and np.all(space.low == space.low.flat[0]):
        scalar["minimum"] = space.low.flat[0].item()
    if np.all(np.isfinite(space.high)) and np.all(space.high == space.high.flat[0]):
        scalar["maximum"] = space.high.flat[0].item()
    return _array_schema(scalar, cast(tuple[int, ...], space.shape))


@to_json_schema.register
def _(space: MultiBinary) -> JSONSchema:
    return _array_schema(
        {"type": "integer", "minimum": 0, "maximum": 1},
        cast(tuple[int, ...], space.shape),
    )


@to_json_schema.register
def _(space: MultiDiscrete) -> JSONSchema:
    scalar: JSONSchema = {
        "type": "integer",
        "minimum": int(np.min(space.start)),
        "maximum": int(np.max(space.start + space.nvec - 1)),
    }
    return _array_schema(scalar, cast(tuple[int, ...], space.shape))


@to_json_schema.register
def _(space: Text) -> JSONSchema:
    return {
        "type": "string",
        "minLength": int(space.min_length),
        "maxLength": int(space.max_length),
    }


@to_json_schema.register
def _(space: Dict) -> JSONSchema:
    properties = {
        key: to_json_schema(child) for key, child in space.spaces.items()
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@to_json_schema.register
def _(space: Tuple) -> JSONSchema:
    variants = [to_json_schema(child) for child in space.spaces]
    item_schema: JSONSchema = variants[0] if len(variants) == 1 else {"anyOf": variants}
    return {
        "type": "array",
        "items": item_schema,
        "minItems": len(variants),
        "maxItems": len(variants),
    }


@to_json_schema.register
def _(space: Sequence) -> JSONSchema:
    return {
        "type": "array",
        "items": to_json_schema(space.feature_space),
    }


@to_json_schema.register
def _(space: PydanticSpace) -> JSONSchema:
    return cast(JSONSchema, space.adapter.json_schema())


__all__ = [
    "JSONSchema",
    "JSONValue",
    "from_json_value",
    "to_json_schema",
    "to_json_value",
]
