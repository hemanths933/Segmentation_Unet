from abc import abstractproperty,abstractmethod,ABC

class Metric(ABC):
    def __init__(self):
        print("Metrics init")
    
    @abstractmethod
    def calculate(y_hat,y):
        pass
