from torch.nn import Module
from torch.serialization import load
from torch.utils.data import DataLoader
from torch import optim,no_grad
from tqdm import tqdm
from utils.logger_module import get_logger

logger = get_logger("trainer")

def train(model:Module,epochs:int,train_dataloader:DataLoader,eval_dataloader:DataLoader,optimizer:optim,loss_fn):
    """
    A function which trains model, epochs times on train_dataloader
    args : model(torch.nn.Module) : A Torch Model.
           epochs(int) : Number of episodes while training.
           train_dataloader(torch.utils.data.DataLoader) : Training DataLoader
    """
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = 0.0
        eval_epoch_loss = 0.0
        for batch in train_dataloader:
            X,y = batch[0],batch[1]
            # Shape(X) : [batch_size,13]  shape(y) : [batch_size,1]
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits,y)
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_epoch_loss = train_epoch_loss/len(train_dataloader)
        for batch in eval_dataloader:
            X,y = batch[0],batch[1]
            with no_grad(): #Weights would be frozen...
                model.eval() # no random attributes...
                logits = model(X)
                loss = loss_fn(logits,y)
                eval_epoch_loss += loss.item()
        eval_epoch_loss = eval_epoch_loss/len(eval_dataloader)
        print(train_epoch_loss,eval_epoch_loss)
    return model

def test_train():
    from data import load_boston_dataset
    from model import Classifier,regression_loss_fn
    train_loader,eval_loader = load_boston_dataset(4,0.9)
    model = Classifier()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    trained_model = train(model,50,train_loader,eval_loader,optimizer,regression_loss_fn)
    pass
if __name__ == "__main__":
    test_train()