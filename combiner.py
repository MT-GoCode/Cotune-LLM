import pandas as pd
from input_data import InputData
import functools

class Combiner:

    def __init__(self, data : InputData):
        self.source = data.source
        self.target = data.target
    
    def naive_merge(self) -> pd.DataFrame:
        return pd.merge(self.target, self.source, right_on='u_userId', left_on='user_id', how='inner')
        
        

    