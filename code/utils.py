import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl


def load_data(directory):
    if(directory=="data1"):
        M_FSM = pd.read_csv(directory + '/pca_l.csv',header=None,index_col=None).values
        D_SSM = pd.read_csv(directory + '/pca_r.csv',header=None,index_col=None).values
        # M_FSM = pd.read_csv(directory + '/DAElig1.csv',header=None,index_col=None).values
        # D_SSM = pd.read_csv(directory + '/DAEr1.csv',header=None,index_col=None).values
        # M_FSM = pd.read_csv(directory + '/LLE_l_300.csv',header=None,index_col=None).values
        # D_SSM = pd.read_csv(directory + '/LLE_r_300.csv',header=None,index_col=None).values
        # M_FSM = pd.read_csv(directory + '/NMF_l_300.csv',header=None,index_col=None).values
        # D_SSM = pd.read_csv(directory + '/NMF_r_300.csv',header=None,index_col=None).values
    if(directory=="data2"):
        M_FSM = pd.read_csv(directory + '/pca_l.csv',header=None,index_col=None).values
        D_SSM = pd.read_csv(directory + '/pca_r.csv',header=None,index_col=None).values
    if(directory=="data3"):
        M_FSM = pd.read_csv(directory + '/pca_l.csv',header=None,index_col=None).values
        D_SSM = pd.read_csv(directory + '/pca_r.csv',header=None,index_col=None).values
    if(directory=="data4"):
        M_FSM = pd.read_csv(directory + '/pca_l.csv',header=None,index_col=None).values
        D_SSM = pd.read_csv(directory + '/pca_r.csv',header=None,index_col=None).values



    ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))
    IM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
                ID[i][j] = D_SSM[i][j]

    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):

                IM[i][j] = M_FSM[i][j]
    return ID, IM

def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/pair.txt', sep=' ',names=['ligand', 'receptor', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


def build_graph(directory, random_seed, ctx):

    ID, IM = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True).to(ctx)
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:ID.shape[0]] = 1
    g.ndata['type'] = node_type

    print('Adding receptor features ...')
    d_data = nd.zeros(shape=(g.number_of_nodes(), ID.shape[1]), dtype='float32', ctx=ctx)
    d_data[: ID.shape[0], :] = nd.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding ligand features ...')
    m_data = nd.zeros(shape=(g.number_of_nodes(), IM.shape[1]), dtype='float32', ctx=ctx)
    m_data[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = nd.from_numpy(IM)
    g.ndata['m_features'] = m_data

    print('Adding edges ...')
    receptor_ids = list(range(0, ID.shape[0]))
    ligand_ids = list(range(0, IM.shape[0]))

    receptor_ids_invmap = {id_: i for i, id_ in enumerate(receptor_ids)}
    ligand_ids_invmap = {id_: i for i, id_ in enumerate(ligand_ids)}

    sample_receptor_vertices = [receptor_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_ligand_vertices = [ligand_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_receptor_vertices, sample_ligand_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_ligand_vertices, sample_receptor_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.readonly()
    print('Successfully build graph !!')

    return g, receptor_ids_invmap, ligand_ids_invmap,ID.shape[0], IM.shape[0]

def build_allgraph(directory,ctx):
    ID, IM = load_data(directory)
    all_associations = pd.read_csv(directory + '/pair.txt', sep=' ', names=['ligand', 'receptor', 'label'])
    samples = all_associations.values
    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True).to(ctx)
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:ID.shape[0]] = 1
    g.ndata['type'] = node_type

    print('Adding receptor features ...')
    d_data = nd.zeros(shape=(g.number_of_nodes(), ID.shape[1]), dtype='float32', ctx=ctx)
    d_data[: ID.shape[0], :] = nd.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding ligand features ...')
    m_data = nd.zeros(shape=(g.number_of_nodes(), IM.shape[1]), dtype='float32', ctx=ctx)
    m_data[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = nd.from_numpy(IM)
    g.ndata['m_features'] = m_data

    print('Adding edges ...')
    receptor_ids = list(range(0, ID.shape[0]))
    ligand_ids = list(range(0, IM.shape[0]))

    receptor_ids_invmap = {id_: i for i, id_ in enumerate(receptor_ids)}
    ligand_ids_invmap = {id_: i for i, id_ in enumerate(ligand_ids)}

    sample_receptor_vertices = [receptor_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_ligand_vertices = [ligand_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_receptor_vertices, sample_ligand_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_ligand_vertices, sample_receptor_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.readonly()
    print('Successfully build all_graph !!')

    return g