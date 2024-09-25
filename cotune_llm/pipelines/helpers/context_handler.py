import yaml
from prefect import context, task
import os
from cotune_llm.pipelines import encode


@task
def initialize_context(config_file: str) -> None:
    with open(config_file) as file:
        config = yaml.safe_load(file)

    pipeline_map = {
        "encode": encode.run,
    }

    results_dir_ = config['data_dir'] + f"results/{config['pipeline']}"
    os.makedirs(results_dir_, exist_ok=True)

    context.results_dir = results_dir_
    context.pipeline = pipeline_map[config["pipeline"]]
    context.options = config["options"]