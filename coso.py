import random

from matplotlib import pyplot as plt

import numpy as np

from skcriteria import Data

a,c = 50, 6


mtx = np.random.rand(a, c)
criteria = np.asarray([random.choice([1,-1]) for n in range(c)])
cnames = ["criteria {}".format(idx) for idx in range(c)]
anames = ["alternative {}".format(idx) for idx in range(a)]
weights = np.random.randint(1, 100, c)
data = Data(mtx, criteria, weights)
data.plot(labelrow=8, mnorm="sum")
plt.show()
