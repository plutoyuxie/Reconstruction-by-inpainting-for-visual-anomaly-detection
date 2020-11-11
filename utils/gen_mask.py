import numpy as np


def gen_mask(k_list, n, im_size):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N**2)]
                tmp = np.asarray(tmp).reshape(N, N)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                Ms.append(tmp)
        yield Ms