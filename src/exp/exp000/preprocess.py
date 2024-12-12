from typing import overload

import polars as pl

from src import utils


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    # --- Add features
    return df


composed_fn = utils.pipe(
    add_features,
)


@overload
def apply_preprocess(df_train: pl.DataFrame, df_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@overload
def apply_preprocess(df_train: pl.DataFrame, df_test: None) -> tuple[pl.DataFrame, None]: ...


def apply_preprocess(df_train: pl.DataFrame, df_test: pl.DataFrame | None) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    df_train = composed_fn(df_train)
    if df_test is not None:
        df_test = composed_fn(df_test)
    return df_train, df_test


# =============================================================================
# Test
# =============================================================================
def _test_apply_preprocess() -> None:
    df_train, df_test = pl.DataFrame(), pl.DataFrame()
    df_train, df_test = apply_preprocess(df_train, df_test)

    print(df_train)
    print(df_test)


if __name__ == "__main__":
    _test_apply_preprocess()
