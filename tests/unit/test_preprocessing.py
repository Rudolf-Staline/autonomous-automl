"""Focused tests for custom fold-local transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autonomous_automl.components import (
    DateFeatureExtractor,
    FrequencyEncoder,
    build_categorical_transformer,
    build_numeric_transformer,
)


def test_frequency_encoder_learns_training_frequencies_and_unknown_zero() -> None:
    train = pd.DataFrame({"city": ["a", "a", "b", None]})
    validation = pd.DataFrame({"city": ["a", "b", "unknown", None]})
    encoder = FrequencyEncoder().fit(train)

    transformed = encoder.transform(validation)

    np.testing.assert_allclose(transformed[:, 0], [0.5, 0.25, 0.0, 0.25])
    assert encoder.get_feature_names_out().tolist() == ["city__frequency"]


def test_date_extractor_uses_explicit_epoch_and_stable_names() -> None:
    frame = pd.DataFrame({"when": ["1970-01-01", "1970-01-02", None]})
    extractor = DateFeatureExtractor().fit(frame)

    transformed = extractor.transform(frame)

    assert transformed[0, -1] == 0.0
    assert transformed[1, -1] == 1.0
    assert np.isnan(transformed[2]).all()
    assert extractor.get_feature_names_out().tolist()[-1] == "when__days_since_1970"


@pytest.mark.parametrize("imputer", ["mean", "median", "constant", "median_indicator", "knn"])
@pytest.mark.parametrize("scaler", ["none", "standard", "robust"])
def test_numeric_strategy_matrix_is_executable(imputer: str, scaler: str) -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, np.inf, 4.0], "y": [2.0, 3.0, 4.0, 5.0]})

    transformed = build_numeric_transformer(imputer, scaler).fit_transform(frame)

    assert transformed.shape[0] == len(frame)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("encoder", ["one_hot", "ordinal", "frequency"])
@pytest.mark.parametrize("imputer", ["constant", "most_frequent"])
def test_categorical_strategy_matrix_handles_unknowns(imputer: str, encoder: str) -> None:
    train = pd.DataFrame({"category": ["a", "b", None, "a"]})
    validation = pd.DataFrame({"category": ["unknown", None]})
    transformer = build_categorical_transformer(imputer, encoder, dense_output=True)

    transformer.fit(train)
    transformed = transformer.transform(validation)

    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (build_numeric_transformer, ("unknown", "none")),
        (build_numeric_transformer, ("median", "unknown")),
    ],
)
def test_unknown_numeric_strategies_fail_with_actionable_errors(
    factory: object,
    args: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_numeric_transformer(*args)
