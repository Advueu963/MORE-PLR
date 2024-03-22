import numpy as np
from _overlap_intervals import test
b = np.random.rand(100,2)
consensus = np.zeros(shape=b.shape[0])
for i in range(100000):
       test(b,consensus.shape[0],consensus)
print(consensus)