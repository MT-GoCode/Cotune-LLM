import pandas as pd
from cotune_llm.utils.constants import day_splits, list_type_columns
from datetime import datetime
from tqdm import tqdm
from cotune_llm.utils.prefect import local_cached_task


@local_cached_task
def naive_merge(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        target,
        source,
        right_on="u_userId",
        left_on="user_id",
        how="inner",
    )


@local_cached_task
def group_time_and_id(
    target: pd.DataFrame, source: pd.DataFrame, full_collapse: bool = True
) -> list[dict]:
    target = add_latent_id(target, "user_id", "pt_d")
    source = add_latent_id(source, "u_userId", "e_et")

    print("Aggregating by user ID and time...")

    target_groups = {
        latent_id: group for latent_id, group in target.groupby("latent_id")
    }
    source_groups = {
        latent_id: group for latent_id, group in source.groupby("latent_id")
    }

    print("Done")

    all_latent_ids = set(target_groups.keys()).union(set(source_groups.keys()))

    if full_collapse:
        # Initialize a list to store the final merged dictionaries
        final_rows = []

        # Iterate over all latent_ids
        for latent_id in tqdm(all_latent_ids, desc="Processing latent_ids"):
            # Collapse the target group if it exists
            collapsed_t = (
                smart_collapser(target_groups[latent_id])
                if latent_id in target_groups
                else None
            )

            # Collapse the source group if it exists
            collapsed_s = (
                smart_collapser(source_groups[latent_id])
                if latent_id in source_groups
                else None
            )

            # Merge the two dictionaries on 'latent_id'
            if collapsed_t is not None and collapsed_s is not None:
                # Merge collapsed_s into collapsed_t
                merged_row = {**collapsed_t, **collapsed_s}  # Merging dictionaries
            elif collapsed_t is not None:
                merged_row = collapsed_t
            elif collapsed_s is not None:
                merged_row = collapsed_s
            else:
                continue

            final_rows.append(merged_row)

        return final_rows


@local_cached_task
def smart_collapser(df, columns_to_exclude=[]):
    summary = {}

    for column in df.columns:
        if column in columns_to_exclude:
            continue
        if column in list_type_columns:
            all_values = (
                df[column]
                .dropna()
                .apply(lambda x: str(x).split("^"))
                .explode()
                .unique()
            )
            summary[column] = list(map(str, all_values))
        else:
            summary[column] = [str(_) for _ in pd.unique(df[column]).tolist()]
    return summary


@local_cached_task
def add_latent_id(df, user_id_col, timestamp_col):
    def assign_time_category(timestamp, splits):
        hour = datetime.strptime(str(timestamp), "%Y%m%d%H%M").hour
        for category, times in splits.items():
            for start, end in times:
                if start <= hour < end:
                    return category
        return "unknown"

    df[timestamp_col] = df[timestamp_col].astype(str)

    df["day"] = df[timestamp_col].str[:8]
    df["day"] = pd.to_datetime(df["day"], format="%Y%m%d").dt.date
    df["time_category"] = df[timestamp_col].apply(
        assign_time_category, args=(day_splits,)
    )
    df["latent_id"] = (
        df[user_id_col].astype(str)
        + "_"
        + df["day"].astype(str)
        + "_"
        + df["time_category"]
    )

    return df
