import torch
import numpy as np
from numba import jit


def fake_data(dtype=torch.float32, batch_size=32, height=224, width=224, depth=3, num_classes=1000):
    image_list = []
    label_list = []
    for _ in range(128):
        image = torch.normal(-1, 1,
                             size=[batch_size, depth, height, width],
                             dtype=dtype)
        label = torch.randint(0, num_classes, size=[1])

        # print(labels)
        image_list.append(image)
        label_list.append(label)

    images = torch.stack(image_list, dim=0)
    labels = torch.stack(label_list, dim=0)

    return images, labels


@ jit(nopython=True)
def xorshift128p(state):
    t = state[0]
    s = state[1]
    state[0] = s
    t ^= t << np.uint64(23)
    t ^= t >> np.uint64(17)
    t ^= s ^ (s >> np.uint64(26))
    state[1] = t
    return np.uint64(t + s)


@ jit(nopython=True)
def bernoulli(p, state):
    t = p * np.iinfo(np.uint64).max
    r = np.array([xorshift128p(state)
                  for _ in range(len(p))], dtype=np.float32)
    return r < t


@ jit(nopython=True)
def randint(low, high, state):
    return xorshift128p(state) % (high - low) + low
