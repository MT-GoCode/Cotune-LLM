import pandas as pd
import functools
from tqdm import tqdm

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

