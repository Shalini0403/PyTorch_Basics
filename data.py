from typing import Tuple
from torch.functional import split
from utils.logger_module import get_logger
from sklearn.datasets import load_boston
from torch import from_numpy
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
logger = get_logger("data_utils")

def load_boston_dataset(batch_size : int,split_ratio : float):
    """
    desc : function for loading boston dataset and split with split_ratio.
    args : batch_size (int) : Size of each batch in the dataset
           split_ratio (float) : Split the Dataset to train/test with the ratio
    """
    X, y = load_boston().data, load_boston().target
    tensor_X,tensor_y = from_numpy(X).float(),from_numpy(y).float().unsqueeze(1)
    dataset = TensorDataset(tensor_X,tensor_y)
    # logger.info(type(tensor_X))
    # logger.info(tensor_X[:2])
    train_idx,test_idx = round(split_ratio*len(tensor_X)),round((1-split_ratio)*len(tensor_X))
    train_dataset,test_dataset = random_split(dataset,[train_idx,test_idx])
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    test_dataloader  = DataLoader(test_dataset,batch_size=batch_size)

    return train_dataloader,test_dataloader

if __name__ == "__main__":
    load_boston_dataset(4,0.9)