import pandas as pd
from datasets import Dataset, DatasetDict
import tempfile
from sklearn.model_selection import train_test_split
from constants import RANDOM_SEED

class Output:
    def __init__(self, encodings : list[str]):
        self.encodings = encodings

    def save_to_txt(self, filename: str, limit : int = None) -> None: # Be sure to set a limit!
        limit = limit if limit else len(self.encodings)
        with open(filename, 'w') as file:
            for line in self.encodings[:limit]:
                file.write(line + '\n')

    def save_to_dataset(self, test_size : float) -> DatasetDict:
        train_data, test_data = train_test_split(self.encodings, test_size=test_size, random_state=RANDOM_SEED)

        # Create datasets from the split data
        train_dataset = Dataset.from_dict({"text": train_data})
        test_dataset = Dataset.from_dict({"text": test_data})

        # Combine into a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        print(dataset_dict)
        return dataset_dict