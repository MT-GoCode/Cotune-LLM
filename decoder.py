import pandas as pd

class Decoder:
    def __init__(self, raw: list[str]):
        self.raw = raw
    
    def GReaT_auto_decode(self):
        return pd.DataFrame([{kv.split(" is ")[0]: kv.split(" is ")[1] for kv in s.split(", ")} for s in self.raw])

