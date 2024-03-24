import time
import random
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics
import torch
#import xgboost as xgb
#import catboost as cat
from hyperparams import hyperparams as params
from net import transNet
from utils import build_graph, sample, load_data,build_allgraph
from model import GNNLRI, GraphEncoder, BilinearDecoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,precision_recall_curve, roc_curve, auc

def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    g, receptor_ids_invmap, ligand_ids_invmap ,ds , ms = build_graph(directory, random_seed=random_seed, ctx=ctx)
    samples = sample(directory, random_seed=random_seed)
    ID, IM = load_data(directory)

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## receptor nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## ligand nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    samples_df = pd.DataFrame(samples, columns=['ligand', 'receptor', 'label'])
    sample_receptor_vertices = [receptor_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_ligand_vertices = [ligand_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    allauc0 = []
    allaupr0 = []
    allacc0 = []
    allpre0 = []
    allrecall0 = []
    allf10 = []

    allauc1 = []
    allaupr1 = []
    allacc1 = []
    allpre1 = []
    allrecall1 = []
    allf11 = []

    allsauc = []
    allsaupr = []
    allsacc = []
    allspre = []
    allsrecall = []
    allsf1 = []


    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    train_index = []
    test_index = []
    for train_idx, test_idx in kf.split(samples[:, 2]):
        train_index.append(train_idx)
        test_index.append(test_idx)


    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        samples_df['train'] = 0
        samples_df['test'] = 0

        samples_df['train'].iloc[train_index[i]] = 1
        samples_df['test'].iloc[test_index[i]] = 1

        train_tensor = nd.from_numpy(samples_df['train'].values.astype('int32')).copyto(ctx)
        test_tensor = nd.from_numpy(samples_df['test'].values.astype('int32')).copyto(ctx)

        edge_data = {'train': train_tensor,
                     'test': test_tensor}

        g.edges[sample_receptor_vertices, sample_ligand_vertices].data.update(edge_data)
        g.edges[sample_ligand_vertices, sample_receptor_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train.copy_from_parent()

        # get the training set
        rating_train = g_train.edata['rating']
        src_train, dst_train = g_train.all_edges()
        # get the testing edge set
        test_eid = g.filter_edges(lambda edges: edges.data['test']).astype('int64')
        src_test, dst_test = g.find_edges(test_eid)
        rating_test = g.edges[test_eid].data['rating']
        src_train = src_train.copyto(ctx)
        src_test = src_test.copyto(ctx)
        dst_train = dst_train.copyto(ctx)
        dst_test = dst_test.copyto(ctx)
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        train_half = int(src_train.asnumpy().shape[0] / 2)
        test_half = int(src_test.asnumpy().shape[0] / 2)
        train_dataset = []
        test_dataset = []
        for m in range(train_half):
            re = src_train.asnumpy()[m]
            li = dst_train.asnumpy()[m]-ds
            lid = samples_df.loc[(samples_df['receptor'] == re) & (samples_df['ligand'] == li)].values.reshape(-1)[0]
            red = samples_df.loc[(samples_df['receptor'] == re)&(samples_df['ligand'] == li)].values.reshape(-1)[1]
            inter = samples_df.loc[(samples_df['receptor'] == re)&(samples_df['ligand'] == li)].values.reshape(-1)[2]
            train_dataset.append(np.hstack((ID[red], IM[lid], inter)))

        train_dataset = pd.DataFrame(train_dataset).values
        feature_train = train_dataset[:,:-1]
        target_train = train_dataset[:,-1].reshape(-1)

        for n in range(test_half):
            re = src_test.asnumpy()[n]
            li = dst_test.asnumpy()[n]-ds
            lid = samples_df.loc[(samples_df['receptor'] == re) & (samples_df['ligand'] == li)].values.reshape(-1)[0]
            red = samples_df.loc[(samples_df['receptor'] == re)&(samples_df['ligand'] == li)].values.reshape(-1)[1]
            inter = samples_df.loc[(samples_df['receptor'] == re)&(samples_df['ligand'] == li)].values.reshape(-1)[2]
            test_dataset.append(np.hstack((ID[red], IM[lid], inter)))

        test_dataset = pd.DataFrame(test_dataset).values
        feature_test = test_dataset[:,:-1]
        target_test = test_dataset[:,-1].reshape(-1)

        model0 = transNet(params.col_num, 256, 1).to(params.device)
        optimizer = torch.optim.Adam(model0.parameters(), lr=params.learning_rate)
        loss_fn = torch.nn.MSELoss().to(params.device)
        for epoch in range(params.epoch_num):
            model0.train()
            feature_train = torch.FloatTensor(feature_train)
            target_train = torch.FloatTensor(target_train)
            train_x = feature_train.to(params.device)
            train_y = target_train.to(params.device)
            pred = model0(train_x)
            loss = loss_fn(pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(loss.item())

        model0.eval()

        feature_test = torch.FloatTensor(feature_test)
        target_test = torch.LongTensor(target_test)
        test_x = feature_test.to(params.device)
        test_y = target_test.to(params.device)
        pred = model0(test_x)
        # score0 = pred.cuda().data.cpu().numpy()
        score0 = pred.data.numpy()


        KT_y_prob_0 = np.arange(0, dtype=float)
        for goal0 in score0:
            KT_y_prob_0 = np.append(KT_y_prob_0, goal0)
        light_y0 = []
        for goal0 in KT_y_prob_0:  # 0 1
            if goal0 > 0.5:
                light_y0.append(1)
            else:
                light_y0.append(0)

        acc0 = accuracy_score(target_test, light_y0)
        precision0 = precision_score(target_test, light_y0)
        recall0 = recall_score(target_test, light_y0)
        f10 = f1_score(target_test, light_y0)

        fpr0, tpr0, thresholds0 = roc_curve(target_test, KT_y_prob_0)
        prec0, rec0, thr0 = precision_recall_curve(target_test, KT_y_prob_0)
        auc0 = auc(fpr0, tpr0)
        aupr0 = auc(rec0, prec0)
        print('--------------------------------------DNN---------------------------------------------')
        print("accuracy:%.4f" % acc0)
        print("precision:%.4f" % precision0)
        print("recall:%.4f" % recall0)
        print("F1 score:%.4f" % f10)
        print("AUC:%.4f" % auc0)
        print("AUPR:%.4f" % aupr0)

        allauc0.append(auc0)
        allaupr0.append(aupr0)
        allacc0.append(acc0)
        allpre0.append(precision0)
        allrecall0.append(recall0)
        allf10.append(f10)



        # Train the model
        model = GNNLRI(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g_train, aggregator=aggregator,
                                    dropout=dropout, slope=slope, ctx=ctx),
                       BilinearDecoder(feature_size=embedding_size))
        model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
        cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

        for epoch in range(epochs):
            start = time.time()
            for _ in range(10):
                with mx.autograd.record():
                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    loss_train.backward()
                trainer.step(1)

            h_val = model.encoder(g)
            score_val = model.decoder(h_val[src_test], h_val[dst_test])
            loss_val = cross_entropy(score_val, rating_test).mean()

            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(score_train.asnumpy()))
            val_auc = metrics.roc_auc_score(np.squeeze(rating_test.asnumpy()), np.squeeze(score_val.asnumpy()))

            results_val = [0 if j < 0.5 else 1 for j in np.squeeze(score_val.asnumpy())]
            accuracy_val = metrics.accuracy_score(rating_test.asnumpy(), results_val)
            precision_val = metrics.precision_score(rating_test.asnumpy(), results_val)
            recall_val = metrics.recall_score(rating_test.asnumpy(), results_val)
            f1_val = metrics.f1_score(rating_test.asnumpy(), results_val)

            end = time.time()

            # print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),
            #       'Val Loss: %.4f' % loss_val.asscalar(),
            #       'Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
            #       'F1: %.4f' % f1_val, 'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
            #       'Time: %.2f' % (end - start))
        h_test = model.encoder(g)
        score_test = model.decoder(h_test[src_test], h_test[dst_test])

        score1 = []
        for s in range (test_half) :
            # print(score_test.asnumpy()[s],'******',score_test.asnumpy()[test_half+s])
            score1.append((score_test.asnumpy()[s]+score_test.asnumpy()[test_half+s])/2)

        score1 = np.array(score1)
        y_ture = np.squeeze(rating_test.asnumpy())[:test_half]

        KT_y_prob_1 = np.arange(0, dtype=float)
        for goal1 in score1:
            KT_y_prob_1 = np.append(KT_y_prob_1, goal1)
        light_y1 = []
        for goal1 in KT_y_prob_1:  # 0 1
            if goal1 > 0.5:
                light_y1.append(1)
            else:
                light_y1.append(0)

        acc1 = accuracy_score(y_ture, light_y1)
        precision1 = precision_score(y_ture, light_y1)
        recall1 = recall_score(y_ture, light_y1)
        f11 = f1_score(y_ture, light_y1)

        fpr1, tpr1, thresholds1 = metrics.roc_curve(y_ture, score1)
        prec1, rec1, thr1 = metrics.precision_recall_curve(y_ture, score1)
        auc1 = metrics.auc(fpr1, tpr1)
        aupr1 = metrics.auc(rec1, prec1)
        print('--------------------------------------HGAE---------------------------------------------')
        print("accuracy:%.4f" % acc1)
        print("precision:%.4f" % precision1)
        print("recall:%.4f" % recall1)
        print("F1 score:%.4f" % f11)
        print("AUC:%.4f" % auc1)
        print("AUPR:%.4f" % aupr1)

        allauc1.append(auc1)
        allaupr1.append(aupr1)
        allacc1.append(acc1)
        allpre1.append(precision1)
        allrecall1.append(recall1)
        allf11.append(f11)


        pred = []
        prob = score0 * 0.1 + score1 * 0.9

        for k in prob:
            if k > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred)

        sacc = accuracy_score(target_test, pred)
        sprecision = precision_score(target_test, pred)
        srecall = recall_score(target_test, pred)
        sf1 = f1_score(target_test, pred)

        sfpr, stpr, sthresholds = roc_curve(target_test, prob)
        sprec, srec, sthr = precision_recall_curve(target_test, prob)
        sauc= auc(sfpr, stpr)
        saupr = auc(srec, sprec)

        print('-------------------------------------CellGDnG---------------------------------------------')
        print("accuracy:%.4f" % sacc)
        print("precision:%.4f" % sprecision)
        print("recall:%.4f" % srecall)
        print("F1 score:%.4f" % sf1)
        print("AUC:%.4f" % sauc)
        print("AUPR:%.4f" % saupr)


        allsauc.append(sauc)
        allsaupr.append(saupr)
        allsacc.append(sacc)
        allspre.append(sprecision)
        allsrecall.append(srecall)
        allsf1.append(sf1)


    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')


    all_g = build_allgraph(directory, ctx=ctx)
    h = model.encoder(all_g)
    src, dst = all_g.all_edges()
    print(src, dst)
    sc = model.decoder(h[src], h[dst])
    src_all = src.asnumpy()
    dst_all = dst.asnumpy()

    all_half = int(src_all.shape[0] / 2)

    all_score1 = []
    for l in range(all_half):
        # print(score_test.asnumpy()[s],'******',score_test.asnumpy()[test_half+s])
        all_score1.append((sc.asnumpy()[l] + sc.asnumpy()[all_half + l]) / 2)

    all_score1 = np.array(all_score1)

    all_associations = pd.read_csv('./data4/pair.txt', sep=' ', names=['ligand', 'receptor', 'label'])
    all_dataset = []
    all_index = []
    for f in range(all_half):
        re = dst_all[f]
        li = src_all[f] - ds
        lid = all_associations.loc[(all_associations['receptor'] == re) & (all_associations['ligand'] == li)].values.reshape(-1)[0]
        red = all_associations.loc[(all_associations['receptor'] == re) & (all_associations['ligand'] == li)].values.reshape(-1)[1]
        inter = all_associations.loc[(all_associations['receptor'] == re) & (all_associations['ligand'] == li)].values.reshape(-1)[2]
        all_index.append([lid,red,inter])
        all_dataset.append(np.hstack((ID[red], IM[lid], inter)))

    all_dataset = pd.DataFrame(all_dataset).values
    all_index = pd.DataFrame(all_index).values
    all_feature = all_dataset[:, :-1]

    model0.eval()
    feature_test = torch.FloatTensor(all_feature)
    test_x = feature_test.to(params.device)
    pred = model0(test_x)
    all_score0 = pred.cuda().data.cpu().numpy()

    all_prob = all_score0 * 0.1 + all_score1 * 0.9

    interMatrix = pd.read_csv("./data4/ligand-receptor interaction.csv", header=0, index_col=0)
    index_r = interMatrix.index.to_list()
    index_c = interMatrix.columns.to_list()

    result = []
    for a in range(int(all_index.shape[0])):
        if (all_index[a,2]==0)&(all_prob[a]>0.9) :
            result.append([index_r[int(all_index[a,0])],index_c[int(all_index[a,1])],all_prob[a]])

    result = pd.DataFrame(data=np.array(result).reshape(-1, 3), columns=['e1', 'e2', 'probability'])
    result.to_csv("./case_study/dataset4_LRI-pred.csv", header=None, index=None)






    # sc = sc.asnumpy()
    # a = []
    # for i in range(len(dst)):
    #     a.append([src_all[i], dst_all[i], sc[i]])
    # b = np.array(a).reshape(-1, 3)
    # c = []
    # length = int(b.shape[0] / 2)
    # for i in range(length):
    #     print(b[i, 0], b[i, 1], '*' * 5, b[length + i, 0], b[length + i, 1])
    #     c.append([b[i, 0], b[i, 1], (b[length + i, 2] + b[i, 2]) / 2])
    # prob = pd.DataFrame(data=np.array(c).reshape(-1, 3), columns=['e1', 'e2', 'probability'])
    # print(prob)
    # prob['e1'] = prob['e1']-int(ID.shape[0])
    # min_probability = 0.8
    # result = prob[prob['probability'] > min_probability]
    # result.to_csv("score.csv",header=None,index=None)

    return   allauc0 ,allaupr0 ,allacc0 ,allpre0 ,allrecall0 ,allf10 ,\
             allauc1 ,allaupr1 ,allacc1 ,allpre1, allrecall1, allf11,\
             allsauc ,allsaupr ,allsacc ,allspre ,allsrecall ,allsf1