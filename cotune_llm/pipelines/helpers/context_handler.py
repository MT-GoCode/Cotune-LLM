import yaml
from prefect import context, task

from cotune_llm.pipelines import encode


@task
def initialize_context(config_file: str) -> None:
    with open(config_file) as file:
        config = yaml.safe_load(file)

    pipeline_map = {
        "encode": encode.run,
    }

    context.pipeline = pipeline_map[config["pipeline"]]
    context.options = config["options"]
