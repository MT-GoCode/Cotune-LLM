from input_data import InputData
from combiner import Combiner
from encoder import Encoder
from output import Output
from constants import *

def generate_GReaT(input_data : InputData) -> list[str]:
    merged_df = Combiner(input_data).naive_merge()
    return Encoder(merged_df).template_textual_encoding_df(GReaT_template_naive)

def generate_GReaT_with_user_time_group(input_data : InputData) -> list[str]:
    people_time_groups = Combiner(input_data).group_time_and_id()
    return Encoder(people_time_groups).template_textual_encoding_dict(GReaT_template_time_user_grouping)
    

def main(combine_and_encode):
    data_container = InputData(Data.TEST_ADS, Data.TEST_FEEDS)
    # data_container.subset(0.05)
    encodings = combine_and_encode(data_container)
    output = Output(encodings=encodings).save_to_dataset(test_size = 0.2)
    # output = Output(encodings=encodings).save_to_txt(filename="throwaway_outputs/test_user_time_group.txt")

if __name__ == "__main__":
    main(generate_GReaT_with_user_time_group)