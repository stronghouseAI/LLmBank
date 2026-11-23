"""A safe calculator for agents: evaluates maths expressions without eval().

Small models are unreliable at arithmetic, so agents hand sums to this instead.
Only numbers, + - * / // % **, brackets and a few named functions are allowed.
"""

from __future__ import annotations

import ast
import math
import operator

MAX_EXPRESSION_LENGTH = 300
MAX_EXPONENT = 1000  # stops "9**9**9" from freezing the app

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}  # fmt: skip
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
}  # fmt: skip
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculationError(ValueError):
    """The expression is not allowed or cannot be worked out."""


def calculate(expression: str) -> float | int:
    """Evaluate a maths expression.

    Raises:
        CalculationError: for anything that is not plain maths, or maths errors
            such as division by zero.
    """
    expression = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")  # noqa: RUF001 - users type these signs
    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError("Give a maths expression of at most 300 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate(tree.body)
    except SyntaxError as error:
        raise CalculationError(f"Not a valid expression: {expression}") from error
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as error:
        raise CalculationError(f"Cannot calculate {expression}: {error}") from error


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"Exponents above {MAX_EXPONENT} are not allowed.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise CalculationError("Only numbers, + - * / % ** and maths functions are allowed.")
