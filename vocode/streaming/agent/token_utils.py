"""Utilities for chat_gpt_agent, mostly around token counting for cost estimation purposes."""

import json
import textwrap
from typing import Any, Dict, List, NamedTuple, Optional

import tiktoken
from loguru import logger

# THE FOLLOWING CODE, UNTIL THE END MARKER, WERE RETRIEVED ON 9/13/2023 FROM
# THE OPENAI COOKBOOK UNDER THE MIT LICENSE.
# MIT License

# Copyright (c) 2023 OpenAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Used to count the amount of tokens Actions add to the billable cost
_FUNCTION_OVERHEAD_STR = """# Tools

## functions

namespace functions {

} // namespace functions"""

CHAT_GPT_MAX_TOKENS = {
    "gpt-3.5-turbo-0613": 4050,
    "gpt-3.5-turbo-16k-0613": 16340,
    "gpt-3.5-turbo-16k": 16340,
    "gpt-3.5-turbo": 16340,
    "gpt-3.5-turbo-1106": 16340,
    "gpt-3.5-turbo-0125": 16340,
    "gpt-4-0314": 8150,
    "gpt-4-32k-0314": 32700,
    "gpt-4-0613": 8150,
    "gpt-4-32k-0613": 32700,
    "gpt-4-0125-preview": 127940,
    "gpt-4-turbo": 127940,
    "gpt-4o": 127940,
    "gpt-4o-2024-05-13": 127940,
}


def get_chat_gpt_max_tokens(model_name: str):
    if model_name.startswith("ft:"):
        model_name = model_name.split(":")[1]

    if model_name in CHAT_GPT_MAX_TOKENS:
        return CHAT_GPT_MAX_TOKENS[model_name]

    return 4050


TokenizerInfo = NamedTuple(
    "TokenizerInfo",
    [
        ("encoding", tiktoken.Encoding),
        ("tokens_per_message", int),
        ("tokens_per_name", int),
    ],
)


def get_tokenizer_info(model: str) -> Optional[TokenizerInfo]:
    if "gpt-35-turbo" in model:
        model = "gpt-3.5-turbo"
    elif "gpt-4o" == model:
        model = "gpt-4o"
    elif "gpt4" in model or "gpt-4" in model:
        model = "gpt-4"
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Warning: model not found. Using cl100k_base encoding for {model}.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.debug(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-4" in model:
        logger.debug(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        tokens_per_message = 3
        tokens_per_name = 1
    elif "llama" in model.lower():
        logger.warning(
            f"Warning: you are using a llama model with an OpenAI compatible endpoint. \
            Llama models are not supported natively support for token counting in tiktoken. \
            Using cl100k_base encoding for {model} as an APPROXIMATION of token usage."
        )
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        return None

    return TokenizerInfo(
        encoding=encoding,
        tokens_per_message=tokens_per_message,
        tokens_per_name=tokens_per_name,
    )


def num_tokens_from_messages(messages: List[dict], model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    tokenizer_info = get_tokenizer_info(model)
    if tokenizer_info is None:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokenizer_info.tokens_per_message
        num_tokens += tokens_from_dict(
            encoding=tokenizer_info.encoding,
            d=message,
            tokens_per_name=tokenizer_info.tokens_per_name,
        )
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# END OF OPENAI COOKBOOK CODE AND GIVEN MIT LICENSE.


def tokens_from_dict(encoding: tiktoken.Encoding, d: Dict[str, Any], tokens_per_name: int) -> int:
    """Return the number of OpenAI tokens in a dict."""
    num_tokens: int = 0
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, str):
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
        elif isinstance(value, dict):
            num_tokens += tokens_from_dict(
                encoding=encoding, d=value, tokens_per_name=tokens_per_name
            )

    return num_tokens


def num_tokens_from_functions(functions: List[dict] | None, model="gpt-3.5-turbo-0613") -> int:
    """Return the number of tokens used by a list of functions."""
    if not functions:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    function_overhead = 3 + len(encoding.encode(_FUNCTION_OVERHEAD_STR))

    return function_overhead + sum(
        len(encoding.encode(_format_func_into_prompt_str(func=f))) for f in functions
    )


# Calculates the amount of tokens added to a given OpenAI prompt for functions
# specifically for billing purposes
def _format_func_into_prompt_str(func) -> str:
    def resolve_ref(schema):
        if schema.get("$ref") is not None:
            ref = schema["$ref"][14:]
            schema = json_schema["definitions"][ref]
        return schema

    def format_schema(schema, indent):
        schema = resolve_ref(schema)
        if "enum" in schema:
            return format_enum(schema, indent)
        elif schema["type"] == "object":
            return format_object(schema, indent)
        elif schema["type"] == "integer":
            return "number"
        elif schema["type"] == "boolean":
            return "boolean"
        elif schema["type"] in ["string", "number"]:
            return schema["type"]
        elif schema["type"] == "array":
            return format_schema(schema["items"], indent) + "[]"
        else:
            raise ValueError("unknown schema type " + schema["type"])

    def format_enum(schema, indent):
        return " | ".join(json.dumps(o) for o in schema["enum"])

    def format_object(schema, indent):
        result = "{\n"
        if "properties" not in schema or len(schema["properties"]) == 0:
            if schema.get("additionalProperties", False):
                return "object"
            return None
        for key, value in schema["properties"].items():
            value = resolve_ref(value)
            value_rendered = format_schema(value, indent + 1)
            if value_rendered is None:
                continue
            if "description" in value and indent == 0:
                for line in textwrap.dedent(value["description"]).strip().split("\n"):
                    result += f"{'  '*indent}// {line}\n"
            optional = "" if key in schema.get("required", {}) else "?"
            comment = (
                "" if value.get("default") is None else f" // default: {format_default(value)}"
            )
            result += f"{'  '*indent}{key}{optional}: {value_rendered},{comment}\n"
        result += ("  " * (indent - 1)) + "}"
        return result

    def format_default(schema):
        v = schema["default"]
        if schema["type"] == "number":
            return f"{v:.1f}" if float(v).is_integer() else str(v)
        else:
            return str(v)

    json_schema = func["parameters"]
    result = f"// {func['description']}\ntype {func['name']} = ("
    formatted = format_object(json_schema, 0)
    if formatted is not None:
        result += "_: " + formatted
    result += ") => any;\n\n"
    return result
