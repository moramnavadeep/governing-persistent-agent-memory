"""Parameter schema extraction for registered tools (Phases C and H).

Phase A found that `ToolSpec.parameters` captured only
`{"required": bool}` -- no types -- so the registry could not emit a
function-calling schema without a rewrite. This module derives typed
parameters from ordinary Python annotations and renders a JSON-Schema /
OpenAI-function-compatible dict.

SCOPE, DELIBERATELY BOUNDED (Phase H)
    Covered: str, int, float, bool, list, dict, Optional[X], X | None,
    list[X] / Sequence[X], dict[str, X], Literal[...] (as an enum), and
    nesting up to three levels inside those.

    NOT covered, on purpose: $ref, allOf/anyOf composition, dataclass /
    TypedDict / Pydantic model introspection, and recursive types. This is
    not an implementation of JSON Schema and does not try to be -- it
    targets what function-calling APIs actually accept and what tool authors
    actually write.

    Anything unrecognized degrades to {"type": "string"} AND is flagged
    `unannotated=True`. That distinction matters: a caller can tell "this
    really is a string" apart from "we could not work out what this is".
    Emitting a confident-looking wrong schema would be worse than admitting
    ignorance, because the model would then be told a parameter is a string
    when the function needs something else.

Deliberate non-goals:
  - No docstring parsing. Extracting per-parameter descriptions from
    free-form docstrings is fragile across styles (Google/NumPy/Sphinx);
    descriptions come from an explicit `param_descriptions=` argument
    instead, so what you get is what you wrote.
  - No MCP implementation. This schema is designed so an MCP or OpenAI
    adapter is mechanical (see aisos/tools/mcp_seam.py), but neither
    protocol is implemented here.
"""
from __future__ import annotations

import collections.abc
import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Python type -> JSON Schema type. Anything unrecognized falls back to
# "string" AND is marked unannotated rather than silently guessed.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_UNSET = inspect.Parameter.empty

#: How deep to follow nested generics before giving up. Three covers
#: dict[str, list[str]] and stops a recursive type alias from looping.
_MAX_NESTING = 3


@dataclass
class ToolParameter:
    """One tool parameter, rendered from a Python annotation.

    `items` / `additional_properties` / `enum` carry the extra structure
    JSON Schema needs for arrays, dicts and Literals. All are optional, so
    a ToolParameter constructed the Phase C way still works.
    """

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    unannotated: bool = False
    #: For type == "array": the schema of each element, e.g. {"type": "string"}.
    items: Optional[dict] = None
    #: For type == "object": the schema of each value, e.g. {"type": "integer"}.
    additional_properties: Optional[dict] = None
    #: For Literal[...]: the permitted values.
    enum: Optional[list] = None

    def to_json_schema(self) -> dict:
        prop: dict[str, Any] = {"type": self.type}
        if self.enum is not None:
            prop["enum"] = list(self.enum)
        if self.items is not None:
            prop["items"] = self.items
        if self.additional_properties is not None:
            prop["additionalProperties"] = self.additional_properties
        if self.description:
            prop["description"] = self.description
        if self.default is not None:
            prop["default"] = self.default
        return prop


def _schema_for(annotation: Any, _depth: int = 0) -> tuple[dict, bool]:
    """Maps a Python annotation to a (JSON-Schema fragment, unannotated) pair."""
    if annotation is _UNSET or annotation is None:
        return {"type": "string"}, True

    if _depth > _MAX_NESTING:
        return {"type": "string"}, True

    try:
        if annotation in _JSON_TYPES:
            return {"type": _JSON_TYPES[annotation]}, False
    except TypeError:  # unhashable annotation (some generic aliases)
        pass

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        # Literal["a", "b"] -> an enum. The JSON type comes from the values
        # themselves; a MIXED-type Literal falls back to an untyped enum
        # rather than asserting a type only some values satisfy.
        values = list(args)
        value_types = {_JSON_TYPES.get(type(v)) for v in values}
        if len(value_types) == 1 and None not in value_types:
            return {"type": value_types.pop(), "enum": values}, False
        return {"type": "string", "enum": values}, True

    # Both typing.Union and PEP 604's `X | None` (types.UnionType).
    if origin is typing.Union or (origin is not None and origin.__class__.__name__ == "UnionType") \
            or str(origin) == "<class 'types.UnionType'>":
        # Optional[X] -> X. Required-ness is decided by the DEFAULT value,
        # never by Optional-ness: a parameter can be Optional[str] and still
        # be required, if it has no default.
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _schema_for(non_none[0], _depth + 1)
        return {"type": "string"}, True

    # Concrete containers AND their collections.abc counterparts:
    # typing.get_origin(Sequence[str]) is collections.abc.Sequence, not list,
    # so checking only the builtins silently degraded every abc-annotated
    # parameter to a bare string. Found by a parametrized schema test.
    if origin in (list, tuple, set, frozenset,
                  collections.abc.Sequence, collections.abc.MutableSequence,
                  collections.abc.Iterable, collections.abc.Collection,
                  collections.abc.Set, collections.abc.MutableSet):
        element_args = [a for a in args if a is not Ellipsis]
        if len(element_args) == 1:
            item_schema, item_unannotated = _schema_for(element_args[0], _depth + 1)
            return {"type": "array", "items": item_schema}, item_unannotated
        return {"type": "array"}, False

    if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
        if len(args) == 2:
            value_schema, value_unannotated = _schema_for(args[1], _depth + 1)
            return {"type": "object", "additionalProperties": value_schema}, value_unannotated
        return {"type": "object"}, False

    return {"type": "string"}, True


def _json_type_for(annotation: Any) -> tuple[str, bool]:
    """Back-compat shim for Phase C callers: (json_type, unannotated),
    without the extra structure. Retained because it was importable."""
    schema, unannotated = _schema_for(annotation)
    return schema["type"], unannotated


def _resolve_hints_individually(fn: Callable[..., Any], signature) -> dict:
    """Best-effort per-parameter annotation resolution.

    Evaluates each annotation string against the function's own globals
    (plus typing), keeping the ones that resolve and dropping the ones that
    do not. Only names the function could already see are evaluated -- this
    resolves annotations, it does not execute caller-supplied code.
    """
    namespace = dict(vars(typing))
    namespace.update(getattr(fn, "__globals__", {}) or {})
    resolved: dict[str, Any] = {}
    for p in signature.parameters.values():
        annotation = p.annotation
        if annotation is _UNSET:
            continue
        if not isinstance(annotation, str):
            resolved[p.name] = annotation
            continue
        try:
            resolved[p.name] = eval(annotation, namespace)  # noqa: S307
        except Exception:
            continue  # genuinely unresolvable -> flagged as unannotated
    return resolved


def extract_parameters(
    fn: Callable[..., Any],
    param_descriptions: Optional[dict[str, str]] = None,
) -> list[ToolParameter]:
    """Build typed ToolParameters from a function's signature.

    *args/**kwargs are skipped: they have no fixed schema, and emitting a
    fake one would be worse than omitting them.
    """
    descriptions = param_descriptions or {}
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        # ONE unresolvable name must not poison every OTHER parameter.
        #
        # typing.get_type_hints() is all-or-nothing: a single name it cannot
        # resolve (a class defined inside a function, a TYPE_CHECKING-only
        # import, a typo) makes the whole call raise. Under PEP 563 / `from
        # __future__ import annotations` every annotation is then a bare
        # string, so falling back to raw annotations degraded EVERY
        # parameter to an unannotated string -- including the plain `str`
        # and `int` ones sitting next to the offending annotation. Found by
        # a test that mixed a locally-defined class with a `str` parameter.
        #
        # Resolve each annotation on its own instead, so only the genuinely
        # unresolvable one is flagged.
        hints = _resolve_hints_individually(fn, signature)

    params: list[ToolParameter] = []
    for p in signature.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.name in ("self", "cls"):
            continue
        annotation = hints.get(p.name, p.annotation)
        schema, unannotated = _schema_for(annotation)
        params.append(
            ToolParameter(
                name=p.name,
                type=schema["type"],
                description=descriptions.get(p.name, ""),
                required=p.default is _UNSET,
                default=None if p.default is _UNSET else p.default,
                unannotated=unannotated,
                items=schema.get("items"),
                additional_properties=schema.get("additionalProperties"),
                enum=schema.get("enum"),
            )
        )
    return params


def parameters_to_json_schema(params: list[ToolParameter]) -> dict:
    """Render an OpenAI-function / JSON-Schema compatible parameter object."""
    return {
        "type": "object",
        "properties": {p.name: p.to_json_schema() for p in params},
        "required": [p.name for p in params if p.required],
    }
