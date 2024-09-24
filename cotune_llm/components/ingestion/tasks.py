import pandas as pd
from cotune_llm.utils.constants import RANDOM_SEED
from cotune_llm.utils.prefect import local_cached_task
from prefect import context


@local_cached_task
def run():
    target = pd.read_csv(context.options["ads_data_path"])
    source = pd.read_csv(context.options["feeds_data_path"])
    target = target.sample(frac=context.options["subset"], random_state=RANDOM_SEED)
    source = source.sample(frac=context.options["subset"], random_state=RANDOM_SEED)

    return target, source
