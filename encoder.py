import pandas as pd
import functools
from tqdm import tqdm

tqdm.pandas()


class Encoder:

    def __init__(self, merged_df : pd.DataFrame):
        self.merged_df = merged_df

    def basic_format_row(self, row, template):
        formatted_row = {k: v for k, v in row.items() if pd.notna(v)}
        return template.format(**formatted_row)

    def GReaT_textual_encoding(self, template) -> list[str]:
        return self.merged_df.progress_apply(lambda row: self.basic_format_row(row, template), axis=1).tolist()