"""Turn the person's run settings into a launch plan plus a forecast of how it will go.

Shared by the Run page, `shai vault load` and background jobs, so all three load a
model exactly the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from stronghouse.core.launch_args import parse_extra_args, streaming_conflicts
from stronghouse.core.models import (
    HardwareProfile,
    LaunchPlan,
    MemorySpeeds,
    ModelProfile,
    PlanSettings,
    QuantOption,
    RunTuning,
    SpeedEstimate,
)
from stronghouse.core.offload_planner import plan_offload
from stronghouse.core.quant_advisor import quant_options, quant_tip
from stronghouse.core.speed_estimator import describe_wait, estimate_speed


@dataclass(frozen=True)
class PlanForecast:
    """What to expect from a plan before loading it."""

    speed: SpeedEstimate
    wait: str
    quant_options: list[QuantOption]
    quant_tip: str | None


def plan_with_settings(
    model: ModelProfile,
    hardware: HardwareProfile,
    settings: PlanSettings,
    default_extra_args: str = "",
) -> LaunchPlan:
    """Auto-fit, then apply any manual overrides and speed flags.

    Raises:
        ValueError: if the context is not positive.
        ExtraArgsError: if the extra flags cannot be read.
    """
    plan = plan_offload(model, hardware, settings.context, settings.kv_cache_type)
    overrides: dict = {}
    if settings.n_cpu_moe is not None:
        overrides["n_cpu_moe"] = min(settings.n_cpu_moe, model.block_count)
    if settings.n_gpu_layers is not None:
        overrides["n_gpu_layers"] = settings.n_gpu_layers
    if overrides:
        plan = replace(plan, notes=[*plan.notes, "Manual override applied."], **overrides)
    extra = default_extra_args if settings.extra_args is None else settings.extra_args
    tuning = RunTuning(
        threads=settings.threads,
        batch_size=settings.batch_size,
        ubatch_size=settings.ubatch_size,
        extra_args=parse_extra_args(extra),
    )
    plan = replace(plan, tuning=tuning)
    conflicts = streaming_conflicts(plan)
    if conflicts:
        note = (
            f"Warning: {' and '.join(conflicts)} loads the whole model into RAM, which is "
            "too small for it. Remove it so the model can stream from disk."
        )
        plan = replace(plan, notes=[*plan.notes, note])
    return plan


def forecast(
    model: ModelProfile,
    plan: LaunchPlan,
    hardware: HardwareProfile,
    filename: str,
    speeds: MemorySpeeds,
) -> PlanForecast:
    """Estimated speed for this plan, and how other quantizations would compare."""
    speed = estimate_speed(model, plan, hardware, speeds)
    options = quant_options(
        model, filename, hardware, plan.context_size, plan.kv_cache_type, speeds
    )
    return PlanForecast(
        speed=speed, wait=describe_wait(speed), quant_options=options, quant_tip=quant_tip(options)
    )
