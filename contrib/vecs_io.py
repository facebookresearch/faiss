# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np

"""
I/O functions in fvecs, bvecs, ivecs formats
definition of the formats here: http://corpus-texmex.irisa.fr/
"""


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def ivecs_mmap(fname):
    assert sys.byteorder != 'big'
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    if sys.byteorder == 'big':
        da = x[:4][::-1].copy()
        d = da.view('int32')[0]
    else:
        d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    if sys.byteorder == 'big':
        m1.byteswap(inplace=True)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))
