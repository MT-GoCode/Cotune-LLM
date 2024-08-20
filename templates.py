from dataclasses import dataclass,field
from functools import cached_property
import pandas as pd

@dataclass
class Template_Factory:
    df : pd.DataFrame
    select_columns : list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.select_columns:  # If select_columns is not provided
            self.select_columns = list(self.df.columns)

    @cached_property
    def GReaT_Template(self):
        columns_no_na = [col for col in self.select_columns if self.df[col].notna().all()]
        return ", ".join([f"{col} is {{{col}}}" for col in columns_no_na])
    
