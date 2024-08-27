import pandas as pd
import functools
from tqdm import tqdm
from constants import list_type_columns

tqdm.pandas()


class Encoder:

    def __init__(self, merged_data : pd.DataFrame | list[dict]):
        self.merged_data = merged_data

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

    def quantize(self) -> pd.Series:
        vocab_dump = {}
        vocab_map = {}

        # Function to generate encoding for each row
        def encode_row(row):
            encode_list = [col for col in list_type_columns if col in self.merged_data.columns]

            # Initialize vocab_dump for each column if not already present
            for col in encode_list:
                if col not in vocab_dump:
                    vocab_dump[col] = []

            list_type_encodings = []
            for col in encode_list:
                values = str(row[col]).split('^')  # Convert row[col] to a string before splitting
                encodings = [f'{col}_{v}' for v in values]
                list_type_encodings.append(' '.join(encodings))
                
                # Add each value to vocab_dump[col]
                vocab_dump[col].extend(encodings)
            
            list_type_encodings_str = ' '.join(list_type_encodings)  # Combine all list-type encodings

            # Process other columns
            other_columns = self.merged_data.columns.difference(list_type_columns)
            other_encodings = [f'{col}_{str(row[col])}' for col in other_columns]
            for col, encoding in zip(other_columns, other_encodings):
                if col not in vocab_dump:
                    vocab_dump[col] = []
                vocab_dump[col].append(encoding)

            # Combine all encodings
            return list_type_encodings_str + ' ' + ' '.join(other_encodings)
        
        # Apply the encoding function to each row of self.merged_data
        encoded_series = self.merged_data.apply(encode_row, axis=1)

        # Reduce vocab_dump[col] to unique values and create vocab_map[col]
        for col, values in vocab_dump.items():
            unique_values = list(set(values))
            vocab_map[col] = {i: v for i, v in enumerate(unique_values)}

        # Return the encoded series
        return encoded_series, vocab_map



