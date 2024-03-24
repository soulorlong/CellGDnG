import torch
#Setting of DNN's hyperparams 
class hyperparams:
    '''rameters'''
    # model
    epoch_num = 300
    col_num = 600
    learning_rate = 0.001
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")