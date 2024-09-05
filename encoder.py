import pandas as pd
import functools
from tqdm import tqdm
from constants import list_type_columns, exclude_columns

tqdm.pandas()


class Encoder:

    def __init__(self, merged_data : pd.DataFrame | list[dict]):
        self.merged_data = merged_data
        self.merged_data.drop(columns = exclude_columns)

    def basic_format_df_row(self, row, template):
        formatted_row = {k: v for k, v in row.items() if pd.notna(v)}
        return template.format(**formatted_row)

    def template_textual_encoding_df(self, template) -> list[str]:
        return self.merged_data.progress_apply(lambda row: self.basic_format_df_row(row, template), axis=1).tolist()
    
    def basic_format_dict(self, row, template):
        formatted_row = {k: ', and '.join(v) for k, v in row.items()}
        r = template['basic'].format(**formatted_row).rstrip()
        if 'user_id' in row:
            r += template['target_template'].format(**formatted_row).rstrip()
        if 'u_userId' in row:
            r += template['source_template'].format(**formatted_row)
        r = r.replace('\n\n', '\n')
        return r

    def template_textual_encoding_dict(self, template) -> list[str]:
        return [self.basic_format_dict(d, template) for d in tqdm(self.merged_data, desc="Encoding dictionaries")]

    def quantize(self):
        vocab_dump = []

        # Function to generate encoding for each row
        def encode_row(row):

            encoded_row_words = [str(row['user_id'])] # start with user id

            row.drop('user_id')
            row.drop('u_userId')

            # STEP 1: QUANTIZE LIST-TYPE COLUMNS

            # encode list is a list of list_type_columns that are included in this row.
            encode_list = [col for col in list_type_columns if col in row.index]

            
            for col in encode_list:
                values = str(row[col]).split('^')  # Convert row[col] to a string before splitting
                encodings = [f'{col}_{v}' for v in values]
                encoded_row_words.extend(encodings)
                vocab_dump.extend(encodings)
            

            # STEP 2: QUANTIZE NON-LIST TYPE COLUMNS
            other_columns = row.index.difference(list_type_columns)
            other_encodings = [f'{col}_{str(row[col])}' for col in other_columns]
            encoded_row_words.extend(other_encodings)
            vocab_dump.extend(other_encodings)

            return ' '.join(encoded_row_words)

        # Apply the encoding function to each row of self.merged_data
        # print(self.merged_data)
        encoded_series = self.merged_data.apply(encode_row, axis=1)

        vocab_dump = list(set(vocab_dump))
        vocab_dump = [str(i) for i in range(10)] + vocab_dump

        # print(encoded_series)
        return encoded_series, vocab_dump



