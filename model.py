from torch.nn import Module,Linear,ReLU
from torch import randn,Tensor
from utils.logger_module import get_logger
from torch.nn import MSELoss

logger = get_logger("model")

class Reg_Mod(Module):
    def __init__(self):
        super(Reg_Mod,self).__init__()
        self.layer_1 = Linear(13,85)
        self.layer_1_2 = Linear(85,100)
        self.layer_2 = Linear(100,1)
        self.act_func = ReLU()
    def forward(self,input):
        """
        args: input (torch.Tensor) : input to be passed
        """
        layered_1 = self.act_func(self.layer_1(input))
        layered_1 = self.act_func(self.layer_1_2(layered_1)) 
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
    Model = Reg_Mod()
    test_logits = Model(test_X)
    logger.info(test_logits)
    assert(test_logits.size() == test_y.size())
    return None

if __name__ == "__main__":
    test_classifier()