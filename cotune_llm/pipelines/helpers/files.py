import os
from datetime import datetime

from prefect import context, task


@task
def create_encode_output_dir(name: str) -> str:
    timestamp = datetime.now().strftime("%m-%d %H-%M")
    path = context.data_dir + "/results/" + name + "_" + timestamp + "/"
    os.makedirs(path)
    return path
