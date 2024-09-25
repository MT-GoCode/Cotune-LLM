from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from cotune_llm.utils.constants import RANDOM_SEED

def save_encode_to_txt(encodings : list[str], limit: int = None) -> None: 
    limit = limit if limit else len(encodings)
    
    context.data
    with open(filename, "w") as file:
        for line in self.encodings[:limit]:
            file.write(line + "\n")

def save_encode_to_dataset(self, test_size: float) -> DatasetDict:
    train_data, test_data = train_test_split(
        self.encodings, test_size=test_size, random_state=RANDOM_SEED
    )

    # Create datasets from the split data
    train_dataset = Dataset.from_dict({"text": train_data})
    test_dataset = Dataset.from_dict({"text": test_data})

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    print(dataset_dict)
    return dataset_dict
