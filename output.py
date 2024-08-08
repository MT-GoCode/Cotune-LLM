import pandas as pd

class Output:
    def __init__(self, encodings : list[str]):
        self.encodings = encodings

    def save_to_txt(self, filename: str):
        with open(filename, 'w') as file:
            for line in self.encodings:
                file.write(line + '\n')