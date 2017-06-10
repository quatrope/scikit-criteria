import random

from matplotlib import pyplot as plt

import numpy as np

from skcriteria import Data

mtx = np.random.rand(10, 10)
criteria = np.asarray([1,-1, 1, 1,1,1,1,1,1,1])
weights = np.random.randint(1, 100, 10)
data = Data(mtx, criteria, weights)
data.plot(mnorm="sum", wnorm="sum", cmap="jet", frame="circle", title="foo")
plt.show()
