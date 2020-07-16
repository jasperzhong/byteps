import unittest

import byteps.mxnet as bps
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import autograd, gluon
from parameterized import parameterized
from tqdm import tqdm

import torch
from utils import bernoulli, fake_data


class QSGDCompressor:
    def __init__(self, quantum_num):
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)
                         ).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(
            torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed


class DitheringTestCase(unittest.TestCase):
    def setUp(self):
        print("init")
        bps.init()

    @parameterized.expand([(2, "linear",), ])
    def test_dithering(self, k, ptype):
        ctx = mx.gpu(0)
        net = get_model("resnet18_v2")
        net.initialize(mx.init.Xavier(), ctx=ctx)
        net.summary(nd.ones((1, 3, 224, 224), ctx=ctx))

        # hyper-params
        seed = 2020
        batch_size = 32
        optimizer_params = {'momentum': 0, 'wd': 0,
                            'learning_rate': 0.01}

        compression_params = {
            "compressor": "dithering",
            "ef": "vanilla",
            # "momentum": "nesterov",
            "k": k,
            "partition": ptype,
            "seed": seed
        }

        trainer = bps.DistributedTrainer(net.collect_params(
        ), "sgd", optimizer_params, compression_params=compression_params)

        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        train_data = fake_data(batch_size=batch_size)

        params = {}
        errors = {}
        errors_s = {}
        moms = {}
        wd_moms = {}
        rngs = {}
        rngs_s = {}
        l2s = {}

        for i, param in enumerate(trainer._params):
            if param.grad_req != 'null':
                params[i] = param._data[0].asnumpy()
                errors[i] = np.zeros_like(params[i])
                errors_s[i] = np.zeros_like(params[i])
                moms[i] = np.zeros_like(params[i])
                wd_moms[i] = np.zeros_like(params[i])
                rngs[i] = np.array([seed, seed], dtype=np.uint64)
                rngs_s[i] = np.array([seed, seed], dtype=np.uint64)
                l2s[i] = []

        for it, batch in tqdm(enumerate(train_data)):
            data = batch[0].as_in_context(ctx)
            label = batch[1].as_in_context(ctx)

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)

            loss.backward()

            gs = {}
            xs = {}

            for i, param in enumerate(trainer._params):
                if param.grad_req != 'null':
                    gs[i] = param._grad[0].asnumpy()
                    xs[i] = param._data[0].asnumpy()

            trainer.step(batch_size)

            for i, param in enumerate(trainer._params):
                if param.grad_req != "null":
                    g = gs[i] / (batch_size * bps.size())
                    # print("norm2", norm2(g.flatten())/k)
                    # moms[i] *= 0.9
                    # moms[i] += g
                    # g += 0.9 * moms[i]
                    g += errors[i]
                    c, l2 = dithering(g, k, rngs[i], ptype)
                    l2s[i].append(l2)
                    errors[i] = g - c

                    # c += errors_s[i]
                    # cs, _ = dithering(c, k, rngs_s[i], ptype)
                    # errors_s[i] = c - cs
                    # c = cs

                    # c += 1e-4*xs[i]
                    params[i] -= optimizer_params["learning_rate"] * c

                    g2 = param._grad[0].asnumpy().flatten()
                    d = c.flatten()
                    if not np.allclose(d, g2, atol=np.finfo(np.float32).eps):
                        print("False")

                        diff = np.abs(d - g2)
                        print(d)  # baseline
                        print(g2)  # byteps
                        print(diff)
                        print(it, i, np.max(diff), np.mean(
                            diff), len(diff), c.shape)
                        idx = np.where(diff > 1e-5)
                        print("g: ", idx, gs[i].flatten()[idx])
                        print("g+e: ", idx, g.flatten()[idx])
                        print("mxnet: ", idx, d[idx])
                        print("byteps: ", idx, g2[idx])
                        # numpy number of close zeros
                        # idx = np.where(np.abs(g.flatten()) < np.finfo(np.float32).eps)
                        # print(len(idx[0]))
                        # print(idx)
                        # # idx = np.where(np.abs(g.flatten()) == 0.0)
                        # # print(len(idx[0]))
                        # # print(idx)
                        input()

        cnt = 0
        tot = 0
        diffs = []
        for i, param in enumerate(trainer._params):
            if param.grad_req != "null":
                x = param._data[0].asnumpy()
                tot += len(x.flatten())
                if not np.allclose(params[i], x, atol=np.finfo(np.float32).eps):
                    print(params[i])
                    print(x)
                    print(np.abs(x-params[i]))
                    diff = np.abs(x.flatten() - params[i].flatten())

                    diffs.append(np.max(diff))
                    idx = np.where(diff > np.finfo(np.float32).eps)
                    cnt += len(idx[0])
                    input()

                plt.plot(l2s[i])
        plt.grid(True)
        plt.savefig("../scripts/pngs/l2-ef-2.png")

        print("false=%d tot=%d false / tot = %lf" % (cnt, tot, cnt / tot))
        if diffs:
            print("max_diff=%f\tmin_diff=%f\tmean_diff=%f" %
                  (np.max(diffs), np.min(diffs), np.mean(diffs)))

        assert cnt == 0


if __name__ == '__main__':
    unittest.main()
