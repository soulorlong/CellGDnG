import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np

from layers import GraphSageLayer


class GNNLRI(nn.Block):
    def __init__(self, encoder, decoder):
        super(GNNLRI, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, ligand, receptor):
        h = self.encoder(G)
        h_ligand = h[ligand]
        h_receptor = h[receptor]
        return self.decoder(h_ligand, h_receptor)


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
        self.ligand_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.receptor_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        for i in range(n_layers):
            if aggregator == 'GraphSAGE':
                self.layers.add(GraphSageLayer(embedding_size, G, self.ligand_nodes, self.receptor_nodes, dropout, slope, ctx))
            else:
                raise NotImplementedError

        self.ligand_emb = LigandEmbedding(embedding_size, dropout)
        self.receptor_emb = ReceptorEmbedding(embedding_size, dropout)

    def forward(self, G):
        # Generate embedding on receptor nodes and ligand nodes
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.apply_nodes(lambda nodes: {'h': self.ligand_emb(nodes.data)}, self.ligand_nodes)
        G.apply_nodes(lambda nodes: {'h': self.receptor_emb(nodes.data)}, self.receptor_nodes)

        for layer in self.layers:
            layer(G)

        return G.ndata['h']


class LigandEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(LigandEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_ligand = seq

    def forward(self, ndata):
        extra_repr = self.proj_ligand(ndata['d_features'])

        return extra_repr


class ReceptorEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(ReceptorEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_receptor = seq

    def forward(self, ndata):
        extra_repr = self.proj_receptor(ndata['m_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_ligand, h_receptor):
        results_mask = self.activation((nd.dot(h_ligand, self.W.data()) * h_receptor).sum(1))

        return results_mask