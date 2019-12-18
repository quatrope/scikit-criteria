
import string

import pytest

import numpy as np

from skcriteria import Data


# =============================================================================
# CONSTANTS
# =============================================================================

CHARS = tuple(string.ascii_letters + string.digits + string.punctuation)


# =============================================================================
# TESTS
# =============================================================================

def test_simple_creation():
    random = np.random.RandomState(42)

    alts = random.randint(2, 10)
    crit = random.randint(2, 10)

    mtx = random.rand(alts, crit)

    criteria = random.choice([min, max], crit)
    weights = random.rand(crit)

    anames = ["".join(random.choice(CHARS, 15)) for _ in range(alts)]
    cnames = ["".join(random.choice(CHARS, 15)) for _ in range(crit)]

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)


    import ipdb; ipdb.set_trace()