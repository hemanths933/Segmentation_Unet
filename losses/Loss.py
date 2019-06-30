from abc import abstractmethod,ABC

class Loss:
    def __init__(self):
        print("loss init")

    @abstractmethod
    def compute_loss(self,yhat,y,weights):
        pass