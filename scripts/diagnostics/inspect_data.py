"""Print a compact diagnostic summary of the processed daily POWER dataset."""

from __future__ import annotations

from wind_scenarios.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
    find_daily_gaps,
    load_power_dataframe,
)


def main() -> None:
    df = load_power_dataframe(DEFAULT_DATA_PATH)
    gaps = find_daily_gaps(df)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nFirst rows:\n", df.head().to_string(index=False))
    print("\nLast rows:\n", df.tail().to_string(index=False))
    print("\nNull values:\n", df.isna().sum())
    print("\nTemporal range:")
    print("Start:", df["DATE"].min())
    print("End:  ", df["DATE"].max())
    print(f"\n{DEFAULT_TARGET_COLUMN} statistics:\n", df[DEFAULT_TARGET_COLUMN].describe())
    print("\nDaily calendar gaps:", len(gaps))
    if not gaps.empty:
        print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
