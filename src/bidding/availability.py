"""Build scenario-varying availability matrices (Q_bar_t^s) for a technology."""

from __future__ import annotations

import numpy as np

from .config import PricesConfig, TechnologyConfig
from .prices import _load_daily_matrix


def static_availability_matrix(tech: TechnologyConfig, S: int) -> np.ndarray:
    """Repeat the static T-length profile identically across all S scenarios."""
    values = np.asarray(tech.availability.values, dtype=float)
    return np.tile(values, (S, 1))


def load_renewable_availability(
    tech: TechnologyConfig,
    prices_cfg: PricesConfig,
    resolution: int,
    scenario_labels: list[str],
) -> np.ndarray:
    """
    Build a (S, T) availability matrix for a non-manageable technology (solar/wind)
    from real ESIOS national generation data, paired day-by-day with the price
    scenarios so the price-resource correlation the thesis relies on is preserved.

    The ESIOS data is aggregate national generation, not per-plant output, so it
    is scaled proportionally: the historical maximum national generation observed
    for that resource within the study window is used as a proxy for the
    "effective installed capacity" the plant's nameplate is a fraction of.
    This approximation is documented as a modelling limitation in the thesis.

    Q_bar_t^s = generation_t^s / max(generation) * nameplate_capacity_mw
    """
    if prices_cfg.renewable_csv_path is None:
        raise ValueError(
            "prices.renewable_csv_path must be set to use "
            "availability.source='renewable_csv'."
        )
    avail_cfg = tech.availability
    value_col = f"{avail_cfg.resource}_mwh"

    gen_matrix, gen_labels = _load_daily_matrix(
        prices_cfg.renewable_csv_path, value_col, resolution, prices_cfg.date_range
    )

    price_dates = set(scenario_labels)
    gen_dates = set(gen_labels)
    if price_dates != gen_dates:
        missing_in_gen = sorted(price_dates - gen_dates)
        missing_in_price = sorted(gen_dates - price_dates)
        raise ValueError(
            "Price and renewable-generation scenario dates do not match — "
            f"{len(missing_in_gen)} date(s) in prices missing from generation CSV "
            f"(e.g. {missing_in_gen[:3]}), "
            f"{len(missing_in_price)} date(s) in generation CSV missing from prices "
            f"(e.g. {missing_in_price[:3]})."
        )

    # Reindex generation rows to match the price matrix's scenario order exactly.
    label_to_row = {label: i for i, label in enumerate(gen_labels)}
    order = [label_to_row[label] for label in scenario_labels]
    gen_aligned = gen_matrix[order, :]

    cap_national = gen_aligned.max()
    if cap_national <= 0:
        raise ValueError(
            f"Historical maximum national generation for resource "
            f"{avail_cfg.resource!r} is non-positive — cannot scale availability."
        )

    return gen_aligned / cap_national * avail_cfg.nameplate_capacity_mw
