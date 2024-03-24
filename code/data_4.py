import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from scipy import interp
from sklearn import metrics
import warnings
import random
from train import Train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    all_auc0 = []
    all_aupr0 = []
    all_acc0 = []
    all_pre0 = []
    all_recall0 = []
    all_f10 = []

    all_auc1 = []
    all_aupr1 = []
    all_acc1 = []
    all_pre1 = []
    all_recall1 = []
    all_f11 = []

    all_sauc = []
    all_saupr = []
    all_sacc = []
    all_spre = []
    all_srecall = []
    all_sf1 = []


    for item in range(1):
        print('第{}/{}次'.format(item+1,20))
        allauc0, allaupr0, allacc0, allpre0, allrecall0, allf10, \
        allauc1, allaupr1, allacc1, allpre1, allrecall1, allf11, \
        allsauc, allsaupr, allsacc, allspre, allsrecall, allsf1 = Train(directory='data4',
                                                      epochs=100,
                                                      aggregator='GraphSAGE',  # 'GraphSAGE'
                                                      embedding_size=128,
                                                      layers=1,
                                                      dropout=0.4,
                                                      slope=0.1,  # LeakyReLU
                                                      lr=0.001,
                                                      wd=1e-3,
                                                      random_seed=int(random.random()),
                                                      ctx=mx.gpu(0))
        all_auc0.append(allauc0)
        all_aupr0.append(allaupr0)
        all_acc0.append(allacc0)
        all_pre0.append(allpre0)
        all_recall0.append(allrecall0)
        all_f10.append(allf10)

        all_auc1.append(allauc1)
        all_aupr1.append(allaupr1)
        all_acc1.append(allacc1)
        all_pre1.append(allpre1)
        all_recall1.append(allrecall1)
        all_f11.append(allf11)

        all_sauc.append(allsauc)
        all_saupr.append(allsaupr)
        all_sacc.append(allsacc)
        all_spre.append(allspre)
        all_srecall.append(allsrecall)
        all_sf1.append(allsf1)

    print('--------------------------------------DNN---------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(all_auc0), np.std(all_auc0)),
          'AUPR mean: %.4f, variance: %.4f \n' % (np.mean(all_aupr0), np.std(all_aupr0)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(all_acc0), np.std(all_acc0)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(all_pre0), np.std(all_pre0)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(all_recall0), np.std(all_recall0)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(all_f10), np.std(all_f10)))

    print('--------------------------------------HGAE---------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(all_auc1), np.std(all_auc1)),
          'AUPR mean: %.4f, variance: %.4f \n' % (np.mean(all_aupr1), np.std(all_aupr1)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(all_acc1), np.std(all_acc1)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(all_pre1), np.std(all_pre1)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(all_recall1), np.std(all_recall1)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(all_f11), np.std(all_f11)))

    print('--------------------------------------CellGDnG---------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(all_sauc), np.std(all_sauc)),
          'AUPR mean: %.4f, variance: %.4f \n' % (np.mean(all_saupr), np.std(all_saupr)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(all_sacc), np.std(all_sacc)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(all_spre), np.std(all_spre)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(all_srecall), np.std(all_srecall)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(all_sf1), np.std(all_sf1)))