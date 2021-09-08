from torch._C import dtype
from data import load_boston_dataset
from model import Reg_Mod,regression_loss_fn
from utils.logger_module import get_logger
from torch.optim import Adam
from train import train_epochs
from utils.seed_fn import set_seed
import argparse

parser = argparse.ArgumentParser(description="Training a Model on Boston Dataset...")

parser.add_argument("batch_size",type=int,help="Batch Size to batch the dataset.")
parser.add_argument("split_ratio",type=float,help="Ratio to split the Dataset..")
parser.add_argument("lr",type=float,help="Learning Rate for the optimizer..")
parser.add_argument("epochs",type=int,help="Number of epochs for the model to be trained..")

#set_seed(42)

args = parser.parse_args()

logger = get_logger("main")


def main(batch_size:int,split_ratio:float,lr:float,epochs:int):
    logger.info("Starting Model Training...")
    trainloader,evalloader = load_boston_dataset(batch_size,split_ratio)
    logger.info("Sucessfully loaded Data into Dataloader...")

    logger.info("Initializing the model...")
    model = Reg_Mod()
    logger.info("Sucessfully initialized model..")

    logger.info("Optimizer init...")
    optimizer = Adam(model.parameters(),lr=lr)
    logger.info("Init Optimizer sucessful...")

    logger.info("Starting to train the model...")
    trained_model = train_epochs(model,epochs,trainloader,evalloader,optimizer,regression_loss_fn)
    logger.info("Sucessfully trained the model..")
    return None

if __name__ == "__main__":
    main(args.batch_size,args.split_ratio,args.lr,args.epochs)