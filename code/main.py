import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from scipy import interp
from sklearn import metrics
import warnings

from train import Train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    auc, aupr, acc, pre, recall, f1, fprs, tprs = Train(directory='data4',
                                                  epochs=120,
                                                  aggregator='GraphSAGE',  # 'GraphSAGE'
                                                  embedding_size=128,
                                                  layers=1,
                                                  dropout=0.4,
                                                  slope=0.1,  # LeakyReLU
                                                  lr=0.001,
                                                  wd=1e-3,
                                                  random_seed=1234,
                                                  ctx=mx.gpu(0))

    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc), np.std(auc)),
          'AUPR mean: %.4f, variance: %.4f \n' % (np.mean(aupr), np.std(aupr)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc), np.std(acc)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre), np.std(pre)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall), np.std(recall)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1), np.std(f1)))