# Copyright 2019 Bytedance Inc. All Rights Reserved.
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
"""Gradient compression algorithms."""

import mxnet as mx
import mxnet.ndarray as nd


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    def compress(self, tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    def decompress(self, tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    def decompress(self, tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class NesterovMomentum(Compressor):
    def __init__(self, compressor, mu):
        self.compressor = compressor
        self.mu = mu
        self.mom = None

    def compress(self, tensor):
        if not self.mom:
            self.mom = nd.zeros_like(tensor)
        self.mom = self.mu * self.mom + tensor
        tensor += self.mu * self.mom
        return self.compressor.compress(tensor)

    def decompress(self, tensor, ctx):
        return self.compressor.decompress(tensor, ctx)


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    def compress(self, tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if 'float' in str(tensor.dtype):
            # Only allow compression from other floating point types
            tensor_compressed = tensor.astype('float16', copy=False)
        return tensor_compressed, tensor.dtype

    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if 'float' in str(dtype):
            tensor_decompressed = tensor.astype(dtype, copy=False)
        return tensor_decompressed


def create_compressor(params):
    compressor = NoneCompressor
    if "byteps_fp16pushpull" in params:
        compressor = FP16Compressor
    if "byteps_momentum_type" in params:
        if params["byteps_momentum_type"] == "nesterov":
            mu = 0.9
            if "byteps_momentum_mu" in params:
                mu = float(params["byteps_momentum_mu"])
            compressor = NesterovMomentum(compressor, mu)

    return compressor
