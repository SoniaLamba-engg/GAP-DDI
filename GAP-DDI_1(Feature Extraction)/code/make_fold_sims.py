"""
Leak-free Stage 1 for GAP-DDI: train the layer-attention GCN separately in each CV fold
on TRAINING associations only, and save per-fold cosine similarity matrices + test indices.

Put this file in  GAP-DDI_1(Feature Extraction)/code/  and run with your original
environment (Python 3.6, TensorFlow 1.12):

    python make_fold_sims.py Fdatasets     (also Cdatasets, MainDataset)

Output:  ../fold_sims/<dataset>/rep<r>_fold<i>_{DrugSim,DiseaseSim,test_ind}.txt
test_ind are 1-based MATLAB linear (column-major) indices into DiDrA (disease x drug),
so main2.m can load them and use exactly the same folds and similarities per fold.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from utils import *
from model import GCNModel
from opt import Optimizer

EPOCHS, EMB, LR, ADJDP, DP, SIMW = 4000, 64, 0.01, 0.6, 0.4, 6
N_FOLDS, N_REPEATS = 10, 10
RAW = "../data_raw"


def train_embeddings(train_drug_dis, drug_sim, dis_sim, seed):
    np.random.seed(seed); tf.reset_default_graph(); tf.set_random_seed(seed)
    adj = sp.csr_matrix(constructHNet(train_drug_dis, drug_sim, dis_sim))
    X = constructNet(train_drug_dis)
    features = sparse_to_tuple(sp.csr_matrix(X))
    adj_orig = sparse_to_tuple(sp.csr_matrix(train_drug_dis.copy()))
    adj_norm = preprocess_graph(adj)
    ph = {'features': tf.sparse_placeholder(tf.float32), 'adj': tf.sparse_placeholder(tf.float32),
          'adj_orig': tf.sparse_placeholder(tf.float32),
          'dropout': tf.placeholder_with_default(0., shape=()),
          'adjdp': tf.placeholder_with_default(0., shape=())}
    model = GCNModel(ph, features[2][1], EMB, features[1].shape[0], adj_norm[1].shape[0],
                     train_drug_dis.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(preds=model.reconstructions,
                        labels=tf.reshape(tf.sparse_tensor_to_dense(ph['adj_orig'], validate_indices=False), [-1]),
                        model=model, lr=LR, num_u=train_drug_dis.shape[0],
                        num_v=train_drug_dis.shape[1], association_nam=train_drug_dis.sum())
    sess = tf.Session(); sess.run(tf.global_variables_initializer())
    fd = {ph['features']: features, ph['adj']: adj_norm, ph['adj_orig']: adj_orig,
          ph['dropout']: DP, ph['adjdp']: ADJDP}
    for _ in range(EPOCHS):
        sess.run(opt.opt_op, feed_dict=fd)
    fd.update({ph['dropout']: 0, ph['adjdp']: 0})
    E, att = sess.run([model.embeddings, model.att], feed_dict=fd)
    sess.close()
    nr = train_drug_dis.shape[0]
    return E[:nr], E[nr:], att


def cosine(E):
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1
    E = E / n
    return E @ E.T


def load(ds):
    if ds == "MainDataset":
        drug_dis = np.loadtxt(f"{RAW}/{ds}/drug_dis.csv", delimiter=",")
        return drug_dis.T, np.loadtxt(f"{RAW}/{ds}/drug_sim.csv", delimiter=","), \
            np.loadtxt(f"{RAW}/{ds}/dis_sim.csv", delimiter=",")
    return (np.loadtxt(f"{RAW}/{ds}/DiDrA.txt"), np.loadtxt(f"{RAW}/{ds}/DrugSim.txt"),
            np.loadtxt(f"{RAW}/{ds}/DiseaseSim.txt"))


if __name__ == "__main__":
    ds = sys.argv[1]
    Y, drug_sim, dis_sim = load(ds)                       # Y: disease x drug (as in main2.m)
    out = f"../fold_sims/{ds}"; os.makedirs(out, exist_ok=True)
    pos = np.flatnonzero(Y.ravel(order="F") == 1)         # MATLAB column-major order
    L = len(pos)
    atts = []
    for r in range(N_REPEATS):
        perm = np.random.RandomState(r).permutation(pos)
        for i in range(1, N_FOLDS + 1):
            test = perm[int(np.floor((i - 1) * L / N_FOLDS)): int(np.floor(i * L / N_FOLDS))]
            y2 = Y.ravel(order="F").copy(); y2[test] = 0; y2 = y2.reshape(Y.shape, order="F")
            HR, HD, att = train_embeddings(np.matrix(y2.T), drug_sim * SIMW, dis_sim * SIMW, seed=100 * r + i)
            atts.append(att)
            p = f"{out}/rep{r + 1}_fold{i}_"
            np.savetxt(p + "DrugSim.txt", cosine(np.asarray(HR)))
            np.savetxt(p + "DiseaseSim.txt", cosine(np.asarray(HD)))
            np.savetxt(p + "test_ind.txt", test + 1, fmt="%d")
            print(ds, "repeat", r + 1, "fold", i, "att", att, flush=True)
    np.savetxt(f"{out}/learned_layer_weights.txt", np.array(atts))
    print("mean learned layer weights:", np.mean(atts, 0))
