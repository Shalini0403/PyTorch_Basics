from torch.nn import Module,Linear,ReLU
from torch import randn,Tensor
from utils.logger_module import get_logger
from torch.nn import MSELoss

logger = get_logger("model")

class Classifier(Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.layer_1 = Linear(13,50)
        self.layer_2 = Linear(50,1)
        self.act_func = ReLU()
    def forward(self,input):
        """
        args: input (torch.Tensor) : input to be passed
        """
        layered_1 = self.act_func(self.layer_1(input))
        layered_2 = self.act_func(self.layer_2(layered_1))
        return layered_2

def regression_loss_fn(logits:Tensor,target:Tensor) -> Tensor:
    """
    Regression Loss Function
    args : logits(torch.Tensor) : Y_Pred
           target(torch.Tensor) : Y_Actual
    """
    return MSELoss()(logits,target)
def test_classifier():
    """
    Testing the Classifier Model...
    """
    test_X,test_y = randn(4,13),randn(4,1)
    Model = Classifier()
    test_logits = Model(test_X)
    logger.info(test_logits)
    assert(test_logits.size() == test_y.size())
    return None

if __name__ == "__main__":
    test_classifier()