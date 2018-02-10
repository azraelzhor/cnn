import numpy as np
import h5py
import matplotlib.pyplot as plt

def conv_forward(X, W, b, params):
    '''
    Arguments:
    X -- input of the convolutional layer (N, Wi, H, C)
    W -- weights need to be learned (F, F, C, Cn)
    b -- biases (Cn,)
    params -- dictionary contains "stride" and "pad" params

    Returns:
    out -- output of the convolution layer (N, Wn, Hn, Cn)
    cache -- cache of values needed for backward pass
    '''
    
    (N, Wi, H, C) = X.shape
    (F, _, _, Cn) = W.shape
    S = params["stride"]
    P = params["pad"]

    Wn = (Wi + 2*P - F)/S + 1
    Hn = (H + 2*P - F)/S + 1

    X_pad = np.pad(X, ((0, 0), (P, P), (P, P), (0, 0)), mode='constant')
    out = np.zeros((N, Wn, Hn, Cn))

    for n in range(N):
        for w in range(Wn):
            for h in range(Hn):
                for c in range(Cn):
                    out[n, w, h, c] = np.sum(X_pad[n, w*S:w*S+F, h*S:h*S+F, :]*W[:, :, :, c]) + b[c]
    
    cache = (X, W, b, params)

    return out, cache

def conv_backward(dout, cache):
    '''
    Arguments:
    dout: gradient of loss w.r.t output of the convolutional layer (N, Wn, Hn, Cn)
    cache: cache of values

    Returns:
    dX: gradient of loss w.r.t input of the convolutional layer (N, Wi, H, C)
    dW: gradient of loss w.r.t weights (F, F, C, Cn)
    db: gradient of loss w.r.t biases (Cn, )
    '''

    (X, W, b, params) = cache
    (N, Wi, H, C) = X.shape
    (N, Wn, Hn, Cn) = dout.shape
    (F, F, C, Cn) = W.shape
    S = params["stride"]
    P = params["pad"]

    # dX out = W*X + b => dX = dout * dout(x)
    dX_pad = np.zeros((N, Wi + 2*P, H + 2*P, C))
    for n in range(N):
        for c in range(Cn):
            for w in range(Wn):
                for h in range(Hn):
                    dX_pad[n, w*S:w*S+F, h*S:h*S+F, :] += dout[n, w, h, c] * W[:, :, :, c]
    dX = dX_pad[:, P:P+Wi, P:P+H, :]

    # dW = dout * dout(W)
    dW = np.zeros(W.shape)
    X_pad = np.pad(X, ((0, 0), (P, P), (P, P), (0, 0)), mode='constant')
    for n in range(N):
        for c in range(Cn):
            for w in range(Wn):
                for h in range(Hn):
                    dW[:, :, :, c] += dout[n, w, h, c] * X_pad[n, w*S:w*S+F, h*S:h*S+F, :]

    # db = dout
    db = np.sum(np.sum(np.sum(dout, axis=0), axis=0), axis=0)

    return dX, dW, db

def pool_forward(X, params, mode="max"):
    pass

def pool_backward(dout, cache, mode="max"):
    pass

def conv_layer_test():
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(8,)
    hparameters = {"pad" : 2,
                "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Actual values:")
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

    print("Expected values:")
    print("Z's mean 	0.0489952035289")
    print("Z[3,2,1] 	[-0.61490741 -6.7439236 -2.55153897 1.75698377 3.56208902 0.53036437 5.18531798 8.75898442]")
    print("cache_conv[0][1][2][3] 	[-0.20075807 0.18656139 0.41005165]")
    
    dA, dW, db = conv_backward(Z, cache_conv)
    print("Actual values:")
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

    print("Expected values:")
    print("dA_mean 	1.45243777754")
    print("dW_mean 	1.72699145831")
    print("db_mean 	7.83923256462") 

if __name__ == "__main__":
    conv_layer_test()
