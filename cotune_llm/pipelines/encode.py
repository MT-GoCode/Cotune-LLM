from cotune_llm.components.encoder import Encoder
from cotune_llm.components.output import Output
from cotune_llm.utils.constants import GReaT_template_time_user_grouping, Data
from cotune_llm.components.templates import Template_Factory
from cotune_llm.components import ingestion, combine
from prefect import flow

@flow
def run():
    target, source = ingestion.run()
    merged_df = combine.naive_merge(target, source)
    return Encoder(merged_df).quantize()

    encodings, vocab_map = combine_and_encode(data_container)
    import json

    with open("vocab_map.json", "w") as json_file:
        json.dump(vocab_map, json_file, indent=4)
    output = Output(encodings=encodings).save_to_dataset(test_size=0.2)
    output = Output(encodings=encodings).save_to_txt(
        filename="throwaway_outputs/test_quantize.txt"
    )


def generate_GReaT(input_data: InputData) -> list[str]:
    merged_df = Combiner(input_data).naive_merge()
    template_factory = Template_Factory(
        merged_df, select_columns=["user_id", "age", "gender"]
    )
    return Encoder(merged_df).template_textual_encoding_df(
        template_factory.GReaT_Template
    )


def generate_GReaT_with_user_time_group(input_data: InputData) -> list[str]:
    people_time_groups = Combiner(input_data).group_time_and_id()
    return Encoder(people_time_groups).template_textual_encoding_dict(
        GReaT_template_time_user_grouping
    )


def generate_quantize(input_data: InputData) -> list[str]:
