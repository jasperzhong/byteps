# Copyright 2020 Amazon Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import reduce

import numpy as np

import mxnet as mx


def get_type_size(dtype):
    if dtype == np.float32:
        return 4
    elif dtype == np.float16 or dtype == np.half:
        return 2
    elif dtype == np.float64:
        return 8
    elif dtype == np.int or dtype == np.int32:
        return 4
    elif dtype == np.int64 or dtype == np.long:
        return 8
    elif dtype == np.int16 or dtype == np.short:
        return 2
    elif dtype == np.int8:
        return 1
    else:
        raise ValueError("unknown dtype: %s" % dtype)


def numel(shape):
    return reduce(lambda x, y: x*y, shape)


def get_tensor_size(tensor):
    return numel(tensor.shape) * get_type_size(tensor.dtype)


class Fusion:
    @staticmethod
    def merge(params):
        grads = []
        ctxs = []
        for param in params:
            grad = param._grad[0]
            ctxs.append(grad.shape)
            grads.append(grad.reshape((-1,)))  # flatten

        return mx.nd.concat(*grads, dim=0), ctxs

    @staticmethod
    def unmerge(merged_grad, ctxs):
        idx = 0
        grads = []
        for ctx in ctxs:
            shape = ctx
            num_ele = numel(shape)
            g = merged_grad[idx:idx+num_ele]
            idx += num_ele
            grads.append(g.reshape(shape))
        return grads
