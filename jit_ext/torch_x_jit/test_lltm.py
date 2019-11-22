import time

import torch
from torch_x_jit.lltm import LLTM

batch_size = 16
input_features = 32
state_size = 128


# Performance on CPU Devices

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))


# Performance on GPU Devices

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

# Note the device=cuda_device arguments here
X = torch.randn(batch_size, input_features, device=cuda_device)
h = torch.randn(batch_size, state_size, device=cuda_device)
C = torch.randn(batch_size, state_size, device=cuda_device)

rnn = LLTM(input_features, state_size).to(cuda_device)

forward = 0
backward = 0
for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    torch.cuda.synchronize()
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    torch.cuda.synchronize()
    backward += time.time() - start

print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))