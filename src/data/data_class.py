import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Any, Tuple
from torch.utils.data import DataLoader, Dataset
from src.data.process_data import make_data

class StockMarketDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, window_size:int, target_column:str='trend') -> None:
        """
        Initializes the StockMarketDataset with specific parameters
        
        Parameters:
        dataframe (pd.DataFrame): The dataset to which contains features and target
        window_size"""
        super().__init__()

        assert dataframe.columns[-1] == target_column, f"The last column of the dataframe should be {target_column}"
        self.features, self.targets = make_data(dataframe, window_size=window_size)

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, index:Any) -> Tuple:
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        return x, y
    