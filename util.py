import numpy
from sklearn import preprocessing
import time
import os
import random
import itertools
import cntk
class Node(object):
    def __init__(self, para, creator, valid, mapping, valid_batch, valid_iter, input_key):
        self.para = para

        network = creator(para)
        temp_err = 0
        for i in range(valid_iter):
            data = valid.next_minibatch(valid_batch, input_map=mapping(valid))
            temp_err += network.test_minibatch(data)
        self.accuracy = 1 - temp_err / valid_iter

        model_name = os.path.join('module', '_'.join(map(str, para)))
        network.model.save(model_name)
        cpu_timer = cntk.load_model(model_name, device=cntk.cpu())

        time_cost = []
        for i in range(valid_iter):
            data = valid.next_minibatch(valid_batch, input_map=mapping(valid))
            arr = numpy.array(data[input_key].as_sequences())
            arr = numpy.reshape(arr, (-1,) + input_key.shape)
            # print arr.shape
            current_time = time.clock()
            cpu_timer.eval(arr, device=cntk.cpu())
            current_time = time.clock() - current_time
            time_cost.append(current_time)
        self.time = numpy.min(time_cost)

    def __str__(self):
        return "{0:10s}\t{1:.2e}ms\t{2:.2f}%".format(','.join(map(str, self.para)), self.time * 1000,
                                                     self.accuracy * 100)


def get_mu(para,hist,hist_perform):

    group_arr = numpy.array(hist)
    scaler = preprocessing.StandardScaler().fit(group_arr.astype(numpy.float64))
    group_arr = scaler.transform(group_arr.astype(numpy.float64))
    Y = numpy.array(hist_perform)
    scaler_T = preprocessing.StandardScaler().fit(Y[:, 1].reshape(-1,1))
    scaler_Y = preprocessing.StandardScaler().fit(Y[:, 0].reshape(-1,1))
    T = scaler_T.transform(Y[:,1].reshape(-1,1)).transpose()
    Y = scaler_Y.transform(Y[:,0].reshape(-1,1)).transpose()
    features = group_arr.shape[1]

    data = numpy.zeros((group_arr.shape[0], 0))
    for i in range(features):
        for j in range(i, features):
            item = group_arr[:, j] * group_arr[:, i]
            item = numpy.reshape(item, (group_arr.shape[0], 1))
            data = numpy.concatenate((data, item), axis=1)
    data = numpy.concatenate((data, numpy.ones((group_arr.shape[0], 1)), group_arr), axis=1)

    w_para = []
    for i in range(features):
        for j in range(i, features):
            w_para.append(para[i] * para[j])
    w_para.append(1)
    w_para = numpy.array(w_para + para)
    w_features = data.shape[0]

    tao = 1e1
    delta = numpy.zeros(w_features)

    for i in range(w_features):
        delta[i] = numpy.exp(-0.5 * numpy.linalg.norm(group_arr[i, :] - numpy.array(para), 2) / tao / tao)

    W = numpy.diag(delta)
    W = W / numpy.sum(W)
    U = numpy.mat(data)
    Y = numpy.mat(Y).transpose()
    T = numpy.mat(T).transpose()
    theta_P = numpy.linalg.pinv(U.transpose() * W * U) * U.transpose() * W * Y
    theta_T = numpy.linalg.pinv(U.transpose() * W * U) * U.transpose() * W * T
    alpha_P = numpy.zeros((features, features))
    alpha_T = numpy.zeros((features, features))
    idx = 0
    for i in range(features):
        for j in range(i, features):
            alpha_P[j, i] = theta_P[idx]
            alpha_T[j, i] = theta_T[idx]
            idx = idx + 1
    idx = idx + 1
    beta_P = numpy.zeros((features, 1))
    beta_T = numpy.zeros((features, 1))
    for i in range(features):
        beta_P[i] = theta_P[idx]
        beta_T[i] = theta_T[idx]

    alpha_P = alpha_P + alpha_P.transpose()
    alpha_T = alpha_T + alpha_T.transpose()
    PP = numpy.array(numpy.mat(alpha_P) * numpy.mat(para).transpose() + beta_P)
    PT = numpy.array(numpy.mat(alpha_T) * numpy.mat(para).transpose() + beta_T)
    mu_P = []
    mu_T = []
    for c in itertools.combinations(range(features), 2):
        mu_P.append(alpha_P[c[0], c[0]] / PP[c[0]] / PP[c[0]])
        mu_P.append(alpha_P[c[0], c[1]] / PP[c[0]] / PP[c[1]])
        mu_P.append(alpha_P[c[1], c[1]] / PP[c[1]] / PP[c[1]])
        mu_T.append(alpha_T[c[0], c[0]] / PT[c[0]] / PP[c[0]])
        mu_T.append(alpha_T[c[0], c[1]] / PT[c[0]] / PP[c[1]])
        mu_T.append(alpha_T[c[1], c[1]] / PT[c[1]] / PP[c[1]])
    muP = max(*mu_P)[0]
    muT = min(*mu_T)[0]
    return muP, muT,scaler_Y,scaler_T,scaler
