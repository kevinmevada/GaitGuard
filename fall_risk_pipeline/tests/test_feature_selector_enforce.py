"""Tests for required-feature enforcement in feature selection."""

from src.features.feature_selector import FeatureSelector


def _selector(max_features: int = 20, max_required: int = 10) -> FeatureSelector:
    return FeatureSelector(
        {
            "paths": {"features": "data/features", "metrics": "results/metrics"},
            "models": {
                "tuning": {"cv_folds": 5},
                "evaluation": {"random_state": 42},
            },
            "feature_selection": {
                "max_features": max_features,
                "max_required_features": max_required,
                "required_feature_substrings": ["sampen", "dfa"],
            },
        }
    )


def test_enforce_required_leaves_rfecv_slots():
    sel = _selector(max_features=20, max_required=10)
    all_features = [f"sampen_feat_{i}" for i in range(12)] + [f"other_{i}" for i in range(5)]
    rfecv_selected = [f"other_{i}" for i in range(5)]
    merged, forced, dropped = sel._enforce_required_features(rfecv_selected, all_features)
    assert len(merged) <= 20
    assert any(f.startswith("other_") for f in merged)
    assert len(forced) == 10


def test_enforce_caps_at_max_features():
    sel = _selector(max_features=10, max_required=10)
    all_features = [f"lb_sampen_{i}" for i in range(15)]
    rfecv_selected = [f"lb_jerk_{i}" for i in range(5)]
    merged, _, _ = sel._enforce_required_features(rfecv_selected, all_features)
    assert len(merged) == 10
    assert all("sampen" in f for f in merged)
