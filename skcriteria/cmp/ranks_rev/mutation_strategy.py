from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class MutationStrategy(ABC):
    @abstractmethod
    def generate_mutations(self, dm: pd.DataFrame, rank: 'Rank') -> pd.DataFrame:
        pass