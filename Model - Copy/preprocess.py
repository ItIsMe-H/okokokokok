import json
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_file):
        # Load data from JSON file
        with open(data_file, 'r') as file:
            self.data = json.load(file)
        
    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve item at index `idx`
        item = self.data[idx]
        command = item['command']
        response = item['response']
        return command, response

# Example usage:
if __name__ == "__main__":
    dataset = TextDataset('data/commands_responses.json')
    print(len(dataset))
    print(dataset[0])
