"""Sanity checks on results/rolling/paths.csv. Run: python tests/test_rolling.py (or pytest)."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PRED_LEN = 20
d = pd.read_csv(BASE / "results/rolling/paths.csv")
HAVE_DATA = (BASE / "data/processed").exists()  # data/ is gitignored; CI only checks result-file consistency


def test_shape_and_finite():
    assert d.groupby("ticker").window.nunique().eq(40).all(), "40 windows per ticker"
    assert d.ticker.nunique() == 30
    vals = d[[c for c in d.columns if c.startswith(("pred_", "act_"))]].to_numpy()
    assert np.isfinite(vals).all() and (vals > 0).all()


def test_actuals_match_processed_data():
    """act_* must equal the 20 closes starting at start_date; last_ctx_close the close just before."""
    if not HAVE_DATA:
        return
    for t, g in d.groupby("ticker"):
        c = pd.read_csv(BASE / f"data/processed/{t}.csv").set_index("date").close
        for _, r in g.sample(3, random_state=0).iterrows():
            i = c.index.get_loc(r.start_date)
            assert np.allclose(c.iloc[i:i + PRED_LEN].to_numpy(), r[[f"act_{k}" for k in range(PRED_LEN)]].to_numpy().astype(float))
            assert np.isclose(c.iloc[i - 1], r.last_ctx_close)


def test_windows_do_not_overlap():
    for t, g in d.groupby("ticker"):
        dates = pd.to_datetime(g.sort_values("window", ascending=False).start_date)
        assert dates.is_monotonic_increasing and dates.is_unique


if __name__ == "__main__":
    for f in (test_shape_and_finite, test_actuals_match_processed_data, test_windows_do_not_overlap):
        f()
        print("ok", f.__name__)
