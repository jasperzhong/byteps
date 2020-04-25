# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import, division, print_function

import itertools
import os

import mxnet as mx
import numpy as np
from mxnet.base import MXNetError
from mxnet.test_utils import same

import byteps.mxnet as bps

has_gpu = mx.context.num_gpus() > 0

# MLSL supports only byte, float and double data types
mlsl_supported_types = set(['float32', 'float64'])


class MXTest:
    """
    Tests for ops in byteps.mxnet.
    """

    def _current_context(self):
        if has_gpu:
            return mx.gpu(bps.local_rank())
        else:
            return mx.current_context()

    def filter_supported_types(self, types):
        if 'MLSL_ROOT' in os.environ:
            types = [t for t in types if t in mlsl_supported_types]
        return types

    def test_byteps_trainer_param_order(self):
        net = mx.gluon.nn.Sequential()
        # layers may be added in a random order for all workers
        layers = {'ones_': 1, 'zeros_': 0}
        for name, init in layers.items():
            net.add(mx.gluon.nn.Dense(10, in_units=10, weight_initializer=mx.init.Constant(init),
                                      use_bias=False, prefix=name))
        params = net.collect_params()
        net.initialize()
        trainer = bps.DistributedTrainer(params, 'sgd')
        trainer._init_params()
        # check the result of bps_broadcast
        for name, init in layers.items():
            weight = params[name + 'weight'].data()[0].asnumpy()
            expected = np.full(shape=weight.shape,
                               fill_value=init, dtype=weight.dtype)
            assert np.array_equal(weight, expected), (weight, expected)

        print('test_byteps_trainer_param_order passed')

    def test_byteps_push_pull(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        dtypes = self.filter_supported_types(['float32'])
        dims = [1]
        ctx = self._current_context()
        count = 100
        shapes = [(), (17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(10 + 10 * bps.rank(), ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)

            print("tensor before push_pull:", tensor)
            bps.byteps_declare_tensor("tensor_pushpull_" + str(count))
            bps.byteps_push_pull(tensor, name="tensor_pushpull_"+str(count))
            tensor.wait_to_read()
            print("tensor after push_pull:", tensor)

        print('test_byteps_push_pull passed')

    def test_byteps_push_pull_inplace(self):
        """Test that the byteps_push_pull correctly sums 1D, 2D, 3D tensors."""
        size = bps.size()
        dtypes = self.filter_supported_types(['int32',   'int64',
                                              'float32', 'float64'])
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 200
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            multiplied = tensor.copy()
            bps.byteps_declare_tensor("tensor_inplace" + str(count))
            bps.byteps_push_pull(tensor, name="tensor_inplace" + str(count))
            max_difference = mx.nd.max(mx.nd.subtract(tensor, multiplied))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            if max_difference > threshold:
                print("self", count, dtype, dim, max_difference, threshold)
                print("tensor", bps.rank(), tensor)
                print("multiplied", bps.rank(), multiplied)
            assert max_difference <= threshold, 'bps.byteps_push_pull produces \
                                                 incorrect results for self'

        print('test_byteps_push_pull_inplace passed')

    def test_byteps_temp_tensor(self):
        """Test that the byteps_push_pull correctly for temporary tensors"""
        dtypes = self.filter_supported_types(['float32'])
        ctxs = [mx.cpu(), mx.gpu(bps.local_rank())]
        expected = mx.nd.array([np.pi, np.e])
        bps.byteps_declare_tensor("tensor_temp")
        for dtype, ctx in itertools.product(dtypes, ctxs):
            tensor = mx.nd.array([np.pi, np.e], ctx=ctx)
            tensor = tensor.astype(dtype)
            bps.byteps_push_pull(tensor, name="tensor_temp")
            tensor.wait_to_read()
            assert same(tensor.asnumpy(), expected.asnumpy()), \
                'bps.byteps_push_pull produces incorrect results for temporary tensors'

        print('test_byteps_temp_tensor passed')


if __name__ == '__main__':
    mxtest = MXTest()
    bps.init()
    mxtest.test_byteps_push_pull()
    mxtest.test_byteps_trainer_param_order()
    mxtest.test_byteps_push_pull_inplace()
    mxtest.test_byteps_temp_tensor()
