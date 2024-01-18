import numpy as np
from os import urandom

def WORD_SIZE():
    return(16)

MASK_VAL = 2 ** WORD_SIZE() - 1;

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def F(x):
    x = ((((rol(x, 1)) & (rol(x, 8)))) & 0xFFFF) ^ rol(x, 2);
    return x;

def expand_key(key, nr):
    Z0 = [0b11111010001001010110000111001101111101000100101011000011100110]
    rk = [0] * nr
    if nr <= 4:
        for i in range(nr):
            rk[i] = key[3-i]
    if nr > 4:
        for i in range(4):
            rk[i] = key[3-i]
        for i in range(4, nr):
            temp = ror(rk[i-1], 3)
            temp = temp ^ rk[i-3]
            temp = temp ^ ror(temp, 1)
            rk[i] = (~(rk[i-4]) ^ temp ^ ((Z0[0] >> (61-(i-4))) & 1) ^ 0x3) & 0xFFFF
    return rk

def enc_one_round(p, k):
    x, y = p[0], p[1]
    y = y ^ (rol(x, 1) & rol(x, 8)) ^ (rol(x, 2)) ^ k
    return (y, x)

def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return (x, y)

def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6565, 0x6877)
    ks = expand_key(key, 32)
    ct = encrypt(pt, ks)
    print("Ciphertext:", ct)
    if (ct == (0xc69b, 0xe9bb)):
        print("Testvector verified.")
        return(True);
    else:
        print("Testvector not verified.")
        return(False)
    
def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def make_train_data(n, nr, diff=(0x0040,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return (X,Y)

#baseline real data generator
def make_real_data(n, nr, diff=(0x0040,0)):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X

#baseline random data generator
def make_random_data(n, nr):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X