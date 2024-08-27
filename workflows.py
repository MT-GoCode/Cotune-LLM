from input_data import InputData
from combiner import Combiner
from encoder import Encoder
from output import Output
from constants import *
from templates import Template_Factory
from decoder import Decoder

def generate_GReaT(input_data : InputData) -> list[str]:
    merged_df = Combiner(input_data).naive_merge()
    template_factory = Template_Factory(merged_df, select_columns=['user_id', 'age', 'gender'])
    return Encoder(merged_df).template_textual_encoding_df(template_factory.GReaT_Template)

def generate_GReaT_with_user_time_group(input_data : InputData) -> list[str]:
    people_time_groups = Combiner(input_data).group_time_and_id()
    return Encoder(people_time_groups).template_textual_encoding_dict(GReaT_template_time_user_grouping)

def generate_quantize(input_data : InputData) -> list[str]:
    merged_df = Combiner(input_data).naive_merge()
    return Encoder(merged_df).quantize()

def encode_workflow(combine_and_encode):
    data_container = InputData(Data.TEST_ADS, Data.TEST_FEEDS)
    data_container.subset(0.05)
    encodings, vocab_map = combine_and_encode(data_container)
    import json
    with open('vocab_map.json', 'w') as json_file:
        json.dump(vocab_map, json_file, indent=4)
    output = Output(encodings=encodings).save_to_dataset(test_size = 0.2)
    output = Output(encodings=encodings).save_to_txt(filename="throwaway_outputs/test_GReaT_dynamic.txt")

def decode_workflow():
    ex = [
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 223611, age is 6, gender is 4",
        "user_id is 133401, age is 7, gender is 2",
        "user_id is 270403, age is 8, gender is 2",
        "user_id is 139353, age is 8, gender is 3",
        "user_id is 195306, age is 5, gender is 2",
    ]
    print(Decoder(ex).GReaT_auto_decode())

if __name__ == "__main__":
    encode_workflow(generate_quantize)
    # decode_workflow()