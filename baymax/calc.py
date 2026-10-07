"""Exact arithmetic, computed in Python instead of guessed by the model.

Small (and not-so-small) local models are unreliable at exact arithmetic -- asked "what's 847
times 39", they'll often produce a confident, wrong number. There's no need to trust a language
model with something a calculator solves exactly, so a plain text gate (same pattern as
baymax.search and baymax.gcal) recognizes an arithmetic question, computes the real answer with
Python's own numbers, and hands that back instead of letting the model estimate.

The expression is evaluated with a hand-rolled walk over the parsed syntax tree, never eval() or
exec(): only numeric literals and a fixed set of arithmetic operators are allowed, so there's no
way for anything -- a name, a call, an import -- to run arbitrary code.
"""

import ast
import operator
import re
from typing import NamedTuple

# Lead-in phrases ("what's", "calculate") and trailing punctuation, stripped before the check.
_LEADING_WORDS = re.compile(r"^\s*(?:what(?:'s|\s+is)|calculate|compute|solve|work\s+out)\s*", re.IGNORECASE)
_TRAILING_PUNCT = re.compile(r"[?.!]+\s*$")

# Spoken operators and the symbols they become. "multiplied by" precedes "times" etc. only for
# readability; none of these phrases overlap, so order doesn't change the result.
_WORD_OPERATORS = [
    (re.compile(r"\bplus\b", re.IGNORECASE), " + "),
    (re.compile(r"\bminus\b", re.IGNORECASE), " - "),
    (re.compile(r"\bmultiplied\s+by\b", re.IGNORECASE), " * "),
    (re.compile(r"\btimes\b", re.IGNORECASE), " * "),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), " / "),
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), " ** "),
    (re.compile(r"\bmod(?:ulo)?\b", re.IGNORECASE), " % "),
]

# Handled before the word operators above, since "of" isn't an arithmetic word on its own.
_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*percent\s+of\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

# The whole normalized question has to be nothing but numbers, operators, and parentheses --
# anything else (a stray word) means this was never a pure arithmetic question to begin with,
# and it's better to leave it to the model than to half-parse a fragment out of a sentence.
_ARITHMETIC_ONLY = re.compile(r"^[\d\s.+\-*/%()]+$")
_HAS_OPERATOR = re.compile(r"[+\-*/%]")

# The only operations _safe_eval will perform, mapped from AST node type to the Python function.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class CalcCheck(NamedTuple):
    """The outcome of checking a question for arithmetic -- shaped just like WebCheck."""

    results: str
    record: str


# Returned when a question isn't arithmetic: nothing to add to the prompt, nothing to remember.
CALC_NOT_CHECKED = CalcCheck("", "")


def extract_expression(text: str) -> str | None:
    """The arithmetic expression ``text`` reduces to, or None if it isn't purely arithmetic."""
    # "15 percent of 80" becomes "(15/100)*80", then spoken operators become symbols.
    normalized = _PERCENT_OF.sub(r"(\1/100)*\2", text)
    for pattern, symbol in _WORD_OPERATORS:
        normalized = pattern.sub(symbol, normalized)
    normalized = _LEADING_WORDS.sub("", normalized)
    normalized = _TRAILING_PUNCT.sub("", normalized).strip()
    # Require an operator too, so a bare number like "42" isn't treated as a calculation.
    if _ARITHMETIC_ONLY.match(normalized) and _HAS_OPERATOR.search(normalized):
        return normalized
    return None


def needs_calculation(text: str) -> bool:
    """Whether ``text`` is a pure arithmetic question."""
    return extract_expression(text) is not None


def lookup(text: str) -> CalcCheck:
    """Compute ``text`` if it's arithmetic, returning prompt context stating the exact answer
    (or an instruction not to guess if it can't be computed); otherwise CALC_NOT_CHECKED."""
    expression = extract_expression(text)
    if expression is None:
        return CALC_NOT_CHECKED
    try:
        value = _safe_eval(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return CalcCheck(
            f"You tried to work out {expression!r} exactly just now, but it didn't come out to a "
            "valid number. Don't state a made-up result; say you're not sure.",
            "You tried to calculate that exactly, but couldn't.",
        )
    result = _format(value)
    return CalcCheck(
        f"The exact result of {expression} is {result}. State this exact number, not an estimate.",
        "You calculated that exactly, using arithmetic, rather than estimating it.",
    )


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a parsed expression, accepting only numbers, + - * / ** % and unary +/-.
    Anything else raises ValueError."""
    # bool is a subclass of int, so it's excluded explicitly.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _format(value: float) -> str:
    """Render a result the way it would be spoken: whole floats as integers, others to 6 places."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)
