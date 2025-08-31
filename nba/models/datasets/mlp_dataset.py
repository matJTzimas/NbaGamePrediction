import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from m

class NBADataset(Dataset):
    def __init__(self, model_config: GeneralModelConfig, split: str):
        """
        Args:

        """
        self.data = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.data)

    def get_train_cols()

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# Example usage
if __name__ == "__main__":
    # dummy data
    features = [[1, 2], [3, 4], [5, 6], [7, 8]]
    labels = [0, 1, 0, 1]

    dataset = NBADataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_features, batch_labels in dataloader:
        print("Features:", batch_features)
        print("Labels:", batch_labels)