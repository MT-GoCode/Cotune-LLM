import pandas as pd

class InputData():

    def __init__(self, target : str, source : str):
        self.target = pd.read_csv(target.value)
        self.source = pd.read_csv(source.value)

    def subset(self, frac, random_state):
        self.target = self.target.sample(frac = frac, random_state = random_state)
        self.source = self.source.sample(frac = frac, random_state = random_state)
