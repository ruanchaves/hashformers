#!/usr/bin/env python3
"""Project hosted-API and Hashformers batch time and cost.

Hashformers throughput and accuracy come from a saved fixed-manifest T4 run.
Hosted API prices and the token, latency, concurrency, and quality scenarios are
explicit inputs. The output is a projection, not an API benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path("benchmarks/qwen/results/2026-08-03-colab-t4-fp16-v3")
DEFAULT_SCENARIO = Path("benchmarks/qwen/hosted_api_cost_scenario.json")
DEFAULT_METADATA = DEFAULT_RESULTS / "hashformers-distilgpt2/run_metadata.json"
DEFAULT_SVG = DEFAULT_RESULTS / "hosted-api-cost-projection.svg"
DEFAULT_COST_SVG = DEFAULT_RESULTS / "hosted-api-total-cost-projection.svg"
DEFAULT_SUMMARY = DEFAULT_RESULTS / "hosted-api-cost-projection.json"


@dataclass(frozen=True)
class HashformersMetrics:
    """Measured Hashformers values used by the projection."""

    label: str
    throughput_items_per_second: float
    exact_match_accuracy: float
    sample_count: int
    repository_revision: str
    metadata_path: str


@dataclass(frozen=True)
class ProviderPrice:
    """One hosted provider's standard token prices."""

    key: str
    label: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    pricing_tier: str
    pricing_source: str


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def require_positive(value: Any, field: str) -> float:
    """Return one finite positive number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return parsed


def load_hashformers_metrics(
    metadata_path: Path = DEFAULT_METADATA,
) -> HashformersMetrics:
    """Load measured wall throughput and proposal accuracy from one run."""

    metadata = read_json(metadata_path)
    if metadata.get("status") != "completed":
        raise ValueError("Hashformers metadata must describe a completed run")
    if metadata.get("repository_dirty") is not False:
        raise ValueError("Hashformers projection requires a clean benchmark run")
    if metadata["measurement"].get("runtime_error_count") != 0:
        raise ValueError("Hashformers projection requires zero runtime errors")

    throughput = require_positive(
        metadata["measurement"]["throughput_items_per_wall_second"],
        "throughput_items_per_wall_second",
    )
    accuracy = require_positive(
        metadata["results"]["overall"]["accuracy"]["rate"],
        "accuracy",
    )
    if accuracy > 1:
        raise ValueError("accuracy must not exceed one")

    try:
        relative_path = metadata_path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        relative_path = metadata_path.resolve()
    return HashformersMetrics(
        label=metadata["model"]["label"],
        throughput_items_per_second=throughput,
        exact_match_accuracy=accuracy,
        sample_count=int(metadata["sample_count"]),
        repository_revision=metadata["repository_revision"],
        metadata_path=relative_path.as_posix(),
    )


def load_providers(scenario: Mapping[str, Any]) -> list[ProviderPrice]:
    """Validate and load hosted provider prices."""

    providers = []
    seen = set()
    for item in scenario["providers"]:
        key = item["key"]
        if key in seen:
            raise ValueError(f"duplicate provider key: {key}")
        seen.add(key)
        providers.append(
            ProviderPrice(
                key=key,
                label=item["label"],
                input_usd_per_million_tokens=require_positive(
                    item["input_usd_per_million_tokens"],
                    f"{key}.input_usd_per_million_tokens",
                ),
                output_usd_per_million_tokens=require_positive(
                    item["output_usd_per_million_tokens"],
                    f"{key}.output_usd_per_million_tokens",
                ),
                pricing_tier=item["pricing_tier"],
                pricing_source=item["pricing_source"],
            )
        )
    if not providers:
        raise ValueError("at least one provider price is required")
    return providers


def hashformers_projection(
    volume: int,
    metrics: HashformersMetrics,
    *,
    t4_hourly_rate: float,
    minimum_billable_seconds: int,
) -> dict[str, float | int]:
    """Project warmed processing time and fresh-job accelerator charge."""

    if volume < 1:
        raise ValueError("volume must be positive")
    processing_seconds = volume / metrics.throughput_items_per_second
    billed_seconds = max(minimum_billable_seconds, math.ceil(processing_seconds))
    return {
        "volume": volume,
        "processing_seconds": processing_seconds,
        "billed_seconds": billed_seconds,
        "cost_usd": billed_seconds * t4_hourly_rate / 3600,
        "cost_per_expected_correct_usd": (
            billed_seconds
            * t4_hourly_rate
            / 3600
            / (metrics.exact_match_accuracy * volume)
        ),
    }


def api_projection(
    volume: int,
    provider: ProviderPrice,
    scenario: Mapping[str, Any],
) -> dict[str, float | int]:
    """Project hosted-API tokens, cost, and scenario wall time."""

    if volume < 1:
        raise ValueError("volume must be positive")
    token_profile = scenario["api_token_profile"]
    quality = scenario["quality_scenario"]
    input_tokens = (
        require_positive(
            token_profile["input_tokens_per_hashtag"],
            "input_tokens_per_hashtag",
        )
        * volume
    )
    output_tokens = (
        require_positive(
            token_profile["output_tokens_per_hashtag"],
            "output_tokens_per_hashtag",
        )
        * volume
    )
    cost = (
        input_tokens * provider.input_usd_per_million_tokens
        + output_tokens * provider.output_usd_per_million_tokens
    ) / 1_000_000
    api_accuracy = require_positive(
        quality["hosted_api_exact_match_accuracy"],
        "hosted_api_exact_match_accuracy",
    )
    if api_accuracy > 1:
        raise ValueError("hosted API accuracy assumption must not exceed one")

    return {
        "volume": volume,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "processing_seconds": api_scenario_seconds(volume, scenario),
        "cost_usd": cost,
        "cost_per_expected_correct_usd": cost / (api_accuracy * volume),
    }


def api_scenario_seconds(volume: int, scenario: Mapping[str, Any]) -> float:
    """Project API wall time from batching, concurrency, and output speed."""

    if volume < 1:
        raise ValueError("volume must be positive")
    profile = scenario["api_token_profile"]
    timing = scenario["api_time_scenario"]
    batch_size = int(require_positive(profile["batch_size"], "batch_size"))
    concurrency = int(
        require_positive(timing["concurrent_requests"], "concurrent_requests")
    )
    fixed_latency = require_positive(
        timing["fixed_request_latency_seconds"],
        "fixed_request_latency_seconds",
    )
    tokens_per_second = require_positive(
        timing["output_tokens_per_second_per_request"],
        "output_tokens_per_second_per_request",
    )
    output_tokens_per_item = require_positive(
        profile["output_tokens_per_hashtag"],
        "output_tokens_per_hashtag",
    )

    def request_seconds(items: int) -> float:
        return fixed_latency + items * output_tokens_per_item / tokens_per_second

    full_requests, partial_items = divmod(volume, batch_size)
    full_latency = request_seconds(batch_size)
    complete_waves, remaining_full_requests = divmod(full_requests, concurrency)
    elapsed = complete_waves * full_latency
    if remaining_full_requests:
        elapsed += full_latency
    if partial_items and remaining_full_requests == 0:
        elapsed += request_seconds(partial_items)
    return elapsed


def logarithmic_volumes(max_volume: int, points: int = 161) -> list[int]:
    """Build unique integer volumes for smooth log-scale curves."""

    if max_volume < 1:
        raise ValueError("max_volume must be positive")
    exponent = math.log10(max_volume)
    values = {
        max(1, round(10 ** (exponent * index / (points - 1))))
        for index in range(points)
    }
    values.add(max_volume)
    return sorted(values)


def first_crossover(
    left,
    right,
    *,
    max_volume: int,
) -> int | None:
    """Find the first integer volume where ``left`` is strictly lower."""

    for volume in range(1, max_volume + 1):
        if left(volume) < right(volume):
            return volume
    return None


def build_projection(
    scenario: Mapping[str, Any],
    metrics: HashformersMetrics,
    *,
    max_volume: int,
) -> dict[str, Any]:
    """Create the auditable projection summary."""

    providers = load_providers(scenario)
    t4 = scenario["t4"]
    hourly_rate = require_positive(t4["hourly_rate"], "t4.hourly_rate")
    minimum_seconds = int(
        require_positive(
            t4["minimum_billable_seconds"],
            "t4.minimum_billable_seconds",
        )
    )
    api_accuracy = require_positive(
        scenario["quality_scenario"]["hosted_api_exact_match_accuracy"],
        "hosted_api_exact_match_accuracy",
    )

    def hash_at(volume: int) -> dict[str, float | int]:
        return hashformers_projection(
            volume,
            metrics,
            t4_hourly_rate=hourly_rate,
            minimum_billable_seconds=minimum_seconds,
        )

    provider_summaries = []
    for provider in providers:
        raw_crossover = first_crossover(
            lambda volume: hash_at(volume)["cost_usd"],
            lambda volume, provider=provider: api_projection(
                volume, provider, scenario
            )["cost_usd"],
            max_volume=max_volume,
        )
        quality_crossover = first_crossover(
            lambda volume: hash_at(volume)["cost_per_expected_correct_usd"],
            lambda volume, provider=provider: api_projection(
                volume, provider, scenario
            )["cost_per_expected_correct_usd"],
            max_volume=max_volume,
        )
        million = api_projection(1_000_000, provider, scenario)
        provider_summaries.append(
            {
                "key": provider.key,
                "label": provider.label,
                "pricing_tier": provider.pricing_tier,
                "pricing_source": provider.pricing_source,
                "input_usd_per_million_tokens": (provider.input_usd_per_million_tokens),
                "output_usd_per_million_tokens": (
                    provider.output_usd_per_million_tokens
                ),
                "projected_cost_per_million_hashtags_usd": million["cost_usd"],
                "projected_cost_per_expected_correct_at_scale_usd": million[
                    "cost_per_expected_correct_usd"
                ],
                "first_volume_hashformers_total_cost_is_lower": raw_crossover,
                "first_volume_hashformers_quality_adjusted_cost_is_lower": (
                    quality_crossover
                ),
            }
        )

    api_time_crossover = first_crossover(
        lambda volume: api_scenario_seconds(volume, scenario),
        lambda volume: hash_at(volume)["processing_seconds"],
        max_volume=max_volume,
    )
    hash_million = hash_at(1_000_000)
    selected = []
    for volume in scenario["projection_volumes"]:
        if volume > max_volume:
            continue
        selected.append(
            {
                "volume": volume,
                "hashformers": hash_at(volume),
                "hosted_api_scenario_seconds": api_scenario_seconds(volume, scenario),
                "provider_costs_usd": {
                    provider.key: api_projection(volume, provider, scenario)["cost_usd"]
                    for provider in providers
                },
            }
        )

    return {
        "schema_version": 1,
        "scenario_id": scenario["scenario_id"],
        "projection_not_measurement": True,
        "pricing_retrieved_date": scenario["pricing_retrieved_date"],
        "currency": scenario["currency"],
        "hashformers": {
            "label": metrics.label,
            "measurement_source": metrics.metadata_path,
            "repository_revision": metrics.repository_revision,
            "sample_count": metrics.sample_count,
            "measured_wall_throughput_items_per_second": (
                metrics.throughput_items_per_second
            ),
            "measured_exact_match_accuracy": metrics.exact_match_accuracy,
            "t4_hourly_rate": hourly_rate,
            "minimum_billable_seconds": minimum_seconds,
            "price_scope": t4["price_scope"],
            "pricing_source": t4["pricing_source"],
            "billing_source": t4["billing_source"],
            "projected_cost_per_million_hashtags_usd": hash_million["cost_usd"],
            "projected_processing_hours_per_million_hashtags": (
                hash_million["processing_seconds"] / 3600
            ),
            "projected_cost_per_expected_correct_at_scale_usd": hash_million[
                "cost_per_expected_correct_usd"
            ],
        },
        "api_token_profile": scenario["api_token_profile"],
        "api_time_scenario": scenario["api_time_scenario"],
        "quality_scenario": scenario["quality_scenario"],
        "providers": provider_summaries,
        "time_projection": {
            "first_volume_hosted_api_scenario_is_faster": api_time_crossover,
            "status": "scenario only; provider latency and rate limits were not measured",
        },
        "selected_volumes": selected,
        "caveats": [
            "Hosted provider exact-match accuracy is an illustrative assumption, not a measured result.",
            "Hosted API wall time is an explicit concurrency and token-speed scenario, not a provider SLA.",
            "Hashformers time excludes model loading and uses warmed measured wall throughput.",
            "The T4 price is accelerator-only; a complete deployment must add VM and operational costs.",
            "Provider token counts use one tokenizer proxy and fixed-manifest averages; actual bills depend on the provider tokenizer, prompt, response, caching, retries, and processing tier.",
        ],
        "max_projected_volume": max_volume,
        "assumed_api_accuracy": api_accuracy,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write stable, readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_plotting() -> tuple[Any, Any]:
    """Import the optional plotting dependency only when rendering."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib: python -m pip install matplotlib"
        ) from exc
    return plt, FuncFormatter


def plot_colors() -> dict[str, str]:
    """Return the stable color mapping shared by both SVG artifacts."""

    return {
        "hashformers": "#0072B2",
        "openai-gpt-5.6-terra": "#009E73",
        "anthropic-claude-haiku-4.5": "#D55E00",
        "google-gemini-3-flash-preview": "#CC79A7",
        "api-time": "#555555",
    }


def configure_plot_style(plt: Any, scenario: Mapping[str, Any]) -> None:
    """Apply deterministic typography and grid styling."""

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.7,
            "svg.fonttype": "none",
            "svg.hashsalt": scenario["scenario_id"],
        }
    )


def draw_total_cost_axis(
    axis: Any,
    *,
    volumes: Sequence[int],
    hash_values: Sequence[Mapping[str, float | int]],
    providers: Sequence[ProviderPrice],
    scenario: Mapping[str, Any],
    metrics: HashformersMetrics,
    summary: Mapping[str, Any],
    colors: Mapping[str, str],
    currency_formatter: Any,
) -> None:
    """Draw the total-inference-cost plot used in both figures."""

    axis.plot(
        volumes,
        [value["cost_usd"] for value in hash_values],
        color=colors["hashformers"],
        linewidth=2.5,
        label=f"{metrics.label}, T4 accelerator",
    )
    for provider_index, (provider, provider_summary) in enumerate(
        zip(providers, summary["providers"], strict=True)
    ):
        values = [
            api_projection(volume, provider, scenario)["cost_usd"] for volume in volumes
        ]
        axis.plot(
            volumes,
            values,
            color=colors[provider.key],
            linewidth=2,
            label=provider.label,
        )
        crossover = provider_summary["first_volume_hashformers_total_cost_is_lower"]
        if crossover is not None:
            crossover_cost = api_projection(crossover, provider, scenario)["cost_usd"]
            axis.scatter(
                [crossover],
                [crossover_cost],
                color=colors[provider.key],
                edgecolor="white",
                linewidth=0.8,
                s=36,
                zorder=5,
            )
            axis.annotate(
                f"{crossover:,}",
                xy=(crossover, crossover_cost),
                xytext=(4, 12 - provider_index * 12),
                textcoords="offset points",
                color=colors[provider.key],
                fontsize=8,
            )
    axis.set_title("Projected total inference cost")
    axis.set_xlabel("Hashtags in one batch")
    axis.set_ylabel("USD")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(currency_formatter)
    axis.legend(loc="upper left", frameon=False)


def save_figure(
    figure: Any,
    plt: Any,
    output: Path,
    scenario: Mapping[str, Any],
) -> None:
    """Save one deterministic plot and normalize Matplotlib SVG whitespace."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.removeprefix(".").lower()
    if output_format not in {"png", "svg"}:
        raise ValueError("plot output must use a .png or .svg extension")
    figure.savefig(
        output,
        format=output_format,
        bbox_inches="tight",
        dpi=180,
        metadata={
            "Date": scenario["pricing_retrieved_date"],
            "Description": (
                "Projection from measured Hashformers T4 throughput and explicit "
                "hosted-API cost, latency, concurrency, and quality assumptions."
            ),
        },
    )
    plt.close(figure)
    if output_format == "svg":
        normalized = "\n".join(
            line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()
        )
        output.write_text(normalized + "\n", encoding="utf-8")


def plot_total_cost_projection(
    output: Path,
    scenario: Mapping[str, Any],
    metrics: HashformersMetrics,
    summary: Mapping[str, Any],
    *,
    max_volume: int,
) -> None:
    """Render the total-inference-cost panel as a standalone SVG."""

    plt, FuncFormatter = load_plotting()
    providers = load_providers(scenario)
    volumes = logarithmic_volumes(max_volume)
    t4 = scenario["t4"]
    hourly_rate = float(t4["hourly_rate"])
    minimum_seconds = int(t4["minimum_billable_seconds"])
    hash_values = [
        hashformers_projection(
            volume,
            metrics,
            t4_hourly_rate=hourly_rate,
            minimum_billable_seconds=minimum_seconds,
        )
        for volume in volumes
    ]

    configure_plot_style(plt, scenario)
    figure, axis = plt.subplots(figsize=(8, 5.4), layout="constrained")
    draw_total_cost_axis(
        axis,
        volumes=volumes,
        hash_values=hash_values,
        providers=providers,
        scenario=scenario,
        metrics=metrics,
        summary=summary,
        colors=plot_colors(),
        currency_formatter=FuncFormatter(format_currency_tick),
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlim(1, max_volume)
    save_figure(figure, plt, output, scenario)


def plot_projection(
    output: Path,
    scenario: Mapping[str, Any],
    metrics: HashformersMetrics,
    summary: Mapping[str, Any],
    *,
    max_volume: int,
) -> None:
    """Render the three-factor projection as a deterministic SVG."""

    plt, FuncFormatter = load_plotting()

    providers = load_providers(scenario)
    volumes = logarithmic_volumes(max_volume)
    t4 = scenario["t4"]
    hourly_rate = float(t4["hourly_rate"])
    minimum_seconds = int(t4["minimum_billable_seconds"])
    api_accuracy = float(
        scenario["quality_scenario"]["hosted_api_exact_match_accuracy"]
    )
    colors = plot_colors()

    hash_values = [
        hashformers_projection(
            volume,
            metrics,
            t4_hourly_rate=hourly_rate,
            minimum_billable_seconds=minimum_seconds,
        )
        for volume in volumes
    ]
    api_time = [api_scenario_seconds(volume, scenario) for volume in volumes]

    configure_plot_style(plt, scenario)
    figure = plt.figure(figsize=(14, 9), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.05))
    time_axis = figure.add_subplot(grid[0, 0])
    cost_axis = figure.add_subplot(grid[0, 1])
    quality_axis = figure.add_subplot(grid[1, :])

    time_axis.plot(
        volumes,
        [value["processing_seconds"] for value in hash_values],
        color=colors["hashformers"],
        linewidth=2.4,
        label=f"{metrics.label}, one warmed T4",
    )
    time_axis.plot(
        volumes,
        api_time,
        color=colors["api-time"],
        linewidth=2.2,
        linestyle="--",
        label="Hosted API scenario, 10 concurrent requests",
    )
    time_crossover = summary["time_projection"][
        "first_volume_hosted_api_scenario_is_faster"
    ]
    if time_crossover is not None:
        time_axis.axvline(
            time_crossover,
            color=colors["api-time"],
            linewidth=1,
            linestyle=":",
        )
        time_axis.text(
            time_crossover * 1.08,
            api_scenario_seconds(time_crossover, scenario) * 1.2,
            f"API scenario first faster\nat {time_crossover:,} hashtags",
            color=colors["api-time"],
            fontsize=8.5,
        )
    time_axis.set_title("Projected elapsed processing time")
    time_axis.set_xlabel("Hashtags in one batch")
    time_axis.set_ylabel("Elapsed time")
    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    time_axis.yaxis.set_major_formatter(FuncFormatter(format_duration_tick))
    time_axis.legend(loc="upper left", frameon=False)

    draw_total_cost_axis(
        cost_axis,
        volumes=volumes,
        hash_values=hash_values,
        providers=providers,
        scenario=scenario,
        metrics=metrics,
        summary=summary,
        colors=colors,
        currency_formatter=FuncFormatter(format_currency_tick),
    )

    quality_axis.plot(
        volumes,
        [value["cost_per_expected_correct_usd"] for value in hash_values],
        color=colors["hashformers"],
        linewidth=2.5,
        label=(f"{metrics.label}: {metrics.exact_match_accuracy:.0%} measured"),
    )
    for provider, provider_summary in zip(providers, summary["providers"], strict=True):
        values = [
            api_projection(volume, provider, scenario)["cost_per_expected_correct_usd"]
            for volume in volumes
        ]
        quality_axis.plot(
            volumes,
            values,
            color=colors[provider.key],
            linewidth=2,
            label=f"{provider.label}: {api_accuracy:.0%} assumed",
        )
        crossover = provider_summary[
            "first_volume_hashformers_quality_adjusted_cost_is_lower"
        ]
        if crossover is not None:
            quality_axis.scatter(
                [crossover],
                [
                    api_projection(crossover, provider, scenario)[
                        "cost_per_expected_correct_usd"
                    ]
                ],
                color=colors[provider.key],
                edgecolor="white",
                linewidth=0.8,
                s=36,
                zorder=5,
            )
            quality_axis.annotate(
                f"{crossover:,}",
                xy=(
                    crossover,
                    api_projection(crossover, provider, scenario)[
                        "cost_per_expected_correct_usd"
                    ],
                ),
                xytext=(4, 5),
                textcoords="offset points",
                color=colors[provider.key],
                fontsize=8,
            )
    quality_axis.set_title(
        "Projected cost per expected correct segmentation "
        "(hosted API quality is hypothetical)"
    )
    quality_axis.set_xlabel("Hashtags in one batch")
    quality_axis.set_ylabel("USD per expected correct result")
    quality_axis.set_xscale("log")
    quality_axis.set_yscale("log")
    quality_axis.yaxis.set_major_formatter(FuncFormatter(format_currency_tick))
    quality_axis.legend(loc="upper right", frameon=False, ncol=2)

    for axis in (time_axis, cost_axis, quality_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(1, max_volume)

    figure.suptitle(
        "Projected batch economics: hosted LLM APIs vs Hashformers on a T4",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "Projection from measured Hashformers-DistilGPT2 T4 throughput "
            f"({metrics.throughput_items_per_second:.2f}/s), $"
            f"{hourly_rate:.2f}/T4-hour with a {minimum_seconds}s minimum, "
            "official standard API token prices, and a 100-item token profile. "
            "Hosted API latency and 90% accuracy are explicit scenarios, not measurements."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    save_figure(figure, plt, output, scenario)


def format_duration_tick(value: float, _position: float) -> str:
    """Format seconds for a log-scale duration axis."""

    if value < 60:
        return f"{compact_number(value)}s"
    if value < 3600:
        return f"{compact_number(value / 60)}m"
    if value < 86400:
        return f"{compact_number(value / 3600)}h"
    return f"{compact_number(value / 86400)}d"


def compact_number(value: float) -> str:
    """Format one chart number with at most one decimal place."""

    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_currency_tick(value: float, _position: float) -> str:
    """Format USD ticks without hiding sub-cent projections."""

    if value < 0.01:
        return f"${value:.3g}"
    if value < 100:
        return f"${value:g}"
    return f"${value:,.0f}"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--hashformers-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST_SVG)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-volume", type=int, default=10_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the projection JSON and SVG artifacts."""

    args = build_parser().parse_args(argv)
    scenario = read_json(args.scenario)
    metrics = load_hashformers_metrics(args.hashformers_metadata)
    summary = build_projection(scenario, metrics, max_volume=args.max_volume)
    write_json(args.summary_output, summary)
    plot_projection(
        args.output,
        scenario,
        metrics,
        summary,
        max_volume=args.max_volume,
    )
    plot_total_cost_projection(
        args.cost_output,
        scenario,
        metrics,
        summary,
        max_volume=args.max_volume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
