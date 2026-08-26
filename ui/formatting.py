"""Value formatting shared by the public app and the admin dashboard.

Presentation only: nothing here loads, scores, queries or decides anything. It
exists because the two apps had already started formatting the same numbers
differently - the bootstrap confidence-interval table was built twice, once with
"95% CI Lower" columns and once with "CI Lower", from the same artifact shape.
One implementation now owns that table, so the two surfaces cannot drift again.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

#: Column labels for the confidence-interval table. Named once so both apps
#: render the same header text.
CI_COLUMNS = ("Metric", "Mean", "95% CI lower", "95% CI upper")


def percent(value: float, decimals: int = 0) -> str:
    """A probability as a percentage: 0.4557 -> '46%'."""
    return f"{value:.{decimals}%}"


def decimal(value: float, places: int = 4) -> str:
    """A metric that is not a percentage, such as ROC-AUC or a Brier score."""
    return f"{value:.{places}f}"


def confidence_interval_table(intervals: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Bootstrap confidence intervals as one table, in artifact order.

    ``intervals`` is the ``confidence_intervals`` mapping written by training,
    keyed by metric name, each value carrying ``mean``, ``ci_lower`` and
    ``ci_upper``. Rows keep the artifact's ordering rather than being sorted, so
    the table reads the same as the file it came from.
    """
    rows = [
        {
            CI_COLUMNS[0]: name.upper(),
            CI_COLUMNS[1]: decimal(values["mean"]),
            CI_COLUMNS[2]: decimal(values["ci_lower"]),
            CI_COLUMNS[3]: decimal(values["ci_upper"]),
        }
        for name, values in intervals.items()
    ]
    return pd.DataFrame(rows, columns=list(CI_COLUMNS))


def count(value: int) -> str:
    """A whole number with thousands separators."""
    return f"{value:,}"
