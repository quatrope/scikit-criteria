from .mutation_strategy import MutationStrategy
import pandas as pd
import numpy as np

class DefaultMutationStrategy(MutationStrategy):
    def generate_mutations(self, dm: pd.DataFrame, rank: 'Rank') -> pd.DataFrame:
        # Example mutation: Add random noise to the decision matrix
        noise = np.random.uniform(-0.1, 0.1, size=dm.shape)
        mutated_dm = dm + noise
        return mutated_dm