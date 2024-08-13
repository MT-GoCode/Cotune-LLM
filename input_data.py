import pandas as pd
from constants import RANDOM_SEED

class InputData():

    def __init__(self, target : str, source : str):
        self.target = pd.read_csv(target.value)
        self.source = pd.read_csv(source.value)

    def subset(self, frac):
        self.target = self.target.sample(frac = frac, random_state = RANDOM_SEED)
        self.source = self.source.sample(frac = frac, random_state = RANDOM_SEED)
