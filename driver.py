from input_data import InputData
from combiner import Combiner
from encoder import Encoder
from output import Output
from constants import Data, GReaT_template

def generate_GReaT(input_data : InputData) -> list[str]:
    template = GReaT_template
    merged_df = Combiner(input_data).naive_merge()
    return Encoder(merged_df).GReaT_textual_encoding(template)

def main():
    data_container = InputData(Data.TEST_ADS, Data.TEST_FEEDS)
    # data_container.subset(0.1, 3)
    encodings = generate_GReaT(data_container)
    output = Output(encodings=encodings).save_to_dataset(test_size=0.2)

if __name__ == "__main__":
    main()