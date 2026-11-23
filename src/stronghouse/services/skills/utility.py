"""Utility skills that touch nothing outside the app, so they run without asking."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from stronghouse.core.calculator import CalculationError, calculate
from stronghouse.services.skills.common import SkillError


def calculator(args: dict) -> str:
    try:
        value = calculate(str(args["expression"]))
    except CalculationError as error:
        raise SkillError(str(error)) from error
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return f"{args['expression']} = {value}"


def current_datetime(_args: dict, now: Callable[[], datetime] = datetime.now) -> str:
    moment = now().astimezone()
    return moment.strftime("%A %d %B %Y, %H:%M (%Z, UTC%z)")
