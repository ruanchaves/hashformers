import pytest

from scripts.segmentation_cost_projection import (
    DEFAULT_COST_SVG,
    DEFAULT_METADATA,
    DEFAULT_SCENARIO,
    DEFAULT_SUMMARY,
    DEFAULT_SVG,
    api_projection,
    api_scenario_seconds,
    build_projection,
    hashformers_projection,
    load_hashformers_metrics,
    load_providers,
    read_json,
)


@pytest.fixture(scope="module")
def scenario():
    return read_json(DEFAULT_SCENARIO)


@pytest.fixture(scope="module")
def metrics():
    return load_hashformers_metrics(DEFAULT_METADATA)


def test_projection_uses_published_adaptive_distilgpt2_measurement(metrics):
    assert metrics.label == "Hashformers-DistilGPT2"
    assert metrics.sample_count == 280
    assert metrics.repository_revision == "d4180e11e383608387685d8f595103adfae8ee72"
    assert metrics.throughput_items_per_second == pytest.approx(14.105325172561523)
    assert metrics.exact_match_accuracy == 0.65


def test_fresh_t4_projection_applies_one_minute_minimum(scenario, metrics):
    t4 = scenario["t4"]
    one = hashformers_projection(
        1,
        metrics,
        t4_hourly_rate=t4["hourly_rate"],
        minimum_billable_seconds=t4["minimum_billable_seconds"],
    )
    million = hashformers_projection(
        1_000_000,
        metrics,
        t4_hourly_rate=t4["hourly_rate"],
        minimum_billable_seconds=t4["minimum_billable_seconds"],
    )

    assert one["billed_seconds"] == 60
    assert one["cost_usd"] == pytest.approx(0.35 / 60)
    assert million["processing_seconds"] == pytest.approx(
        1_000_000 / 14.105325172561523
    )
    assert million["cost_usd"] == pytest.approx(6.892666666666666)


def test_api_cost_uses_explicit_token_projection(scenario):
    providers = {provider.key: provider for provider in load_providers(scenario)}
    openai = api_projection(1_000_000, providers["openai-gpt-5.6-terra"], scenario)
    anthropic = api_projection(
        1_000_000, providers["anthropic-claude-haiku-4.5"], scenario
    )
    google = api_projection(
        1_000_000, providers["google-gemini-3-flash-preview"], scenario
    )

    assert openai["cost_usd"] == pytest.approx(198.99107142857142)
    assert anthropic["cost_usd"] == pytest.approx(68.13214285714285)
    assert google["cost_usd"] == pytest.approx(39.79821428571429)


def test_summary_reports_raw_and_quality_adjusted_crossovers(scenario, metrics):
    summary = build_projection(scenario, metrics, max_volume=10_000_000)
    providers = {provider["key"]: provider for provider in summary["providers"]}

    assert summary["projection_not_measurement"] is True
    assert summary["quality_scenario"]["status"].startswith("illustrative assumption")
    assert (
        providers["openai-gpt-5.6-terra"][
            "first_volume_hashformers_total_cost_is_lower"
        ]
        == 30
    )
    assert (
        providers["anthropic-claude-haiku-4.5"][
            "first_volume_hashformers_total_cost_is_lower"
        ]
        == 86
    )
    assert (
        providers["google-gemini-3-flash-preview"][
            "first_volume_hashformers_total_cost_is_lower"
        ]
        == 147
    )
    assert (
        providers["openai-gpt-5.6-terra"][
            "first_volume_hashformers_quality_adjusted_cost_is_lower"
        ]
        == 41
    )
    assert (
        providers["anthropic-claude-haiku-4.5"][
            "first_volume_hashformers_quality_adjusted_cost_is_lower"
        ]
        == 119
    )
    assert (
        providers["google-gemini-3-flash-preview"][
            "first_volume_hashformers_quality_adjusted_cost_is_lower"
        ]
        == 203
    )
    assert read_json(DEFAULT_SUMMARY) == summary


def test_api_time_projection_exposes_parallel_scenario_crossover(scenario, metrics):
    summary = build_projection(scenario, metrics, max_volume=10_000)

    assert api_scenario_seconds(100, scenario) == pytest.approx(11.964285714285714)
    assert api_scenario_seconds(1000, scenario) == pytest.approx(11.964285714285714)
    assert (
        summary["time_projection"]["first_volume_hosted_api_scenario_is_faster"] == 169
    )


@pytest.mark.parametrize("path", [DEFAULT_SVG, DEFAULT_COST_SVG])
def test_published_svgs_have_no_trailing_whitespace(path):
    svg = path.read_text(encoding="utf-8")

    assert all(line == line.rstrip() for line in svg.splitlines())


def test_standalone_cost_svg_contains_only_total_cost_panel():
    svg = DEFAULT_COST_SVG.read_text(encoding="utf-8")

    assert "Projected total inference cost" in svg
    assert "Projected elapsed processing time" not in svg
    assert "Projected cost per expected correct segmentation" not in svg
