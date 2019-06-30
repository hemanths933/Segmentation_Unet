from models.Unet import Unet
import tensorflow as tf
from losses.Pixelwise_weighted_loss import Pixelwise_weighted_loss
from losses.Dice_loss import Dice_loss
class Import_parameters:
    def __init__(self):
        print("import parameters init")

    def get_parameters(self):
        models = {1: Unet()}
        losses = {1: Pixelwise_weighted_loss().compute_loss,
                  2: Dice_loss().compute_loss}
        optimizers = {1: tf.train.GradientDescentOptimizer}

        prompt_msg = {"model": "Enter the model to execute. Enter 1 for Unet",
                      "loss": "Enter the loss function.  Enter 1 for Weighted loss; 2 for Dice Loss",
                      "optimizer": "Enter the optimizer. Enter 1 for SGD Optimizer",
                      "learning_rate":"Enter the learning rate between 0 and 1",
                      "weights_initializer":"Enter the weight initialization method. Enter 1 for Guassian-paper; 2 for xavier"
                      }

        model_opt = int(input(prompt_msg['model']))
        if model_opt not in models:
            raise Exception("Please enter valid model name")
        else:
            model = models[model_opt]

        loss_opt = int(input(prompt_msg['loss']))
        if loss_opt not in losses:
            raise Exception("Please enter the valid loss function")
        else:
            loss = losses[loss_opt]
            model.loss = loss

        optimizer_opt = int(input(prompt_msg['optimizer']))
        if optimizer_opt not in optimizers:
            raise Exception("Please enter the valid optimizer")
        else:
            optimizer = optimizers[optimizer_opt]

        learning_rate_opt = float(input(prompt_msg['learning_rate']))
        if learning_rate_opt > 0.0 and learning_rate_opt < 1.0:
            lr = tf.train.exponential_decay(learning_rate_opt, tf.Variable(0, trainable=False),
                                           10, 0.8, staircase=True)
            model.optimizer = optimizer(lr)
        else:
            raise Exception("Please enter valid learning rate")
            
        #int(input(prompt_msg['weights_initializer']))

        return model

if __name__=='__main__':
    model = Import_parameters().get_parameters()
    print("sucess")