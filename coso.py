import skcriteria as skc
from skcriteria.preprocessing.invert_objectives import NegateMinimize, InvertMinimize
from skcriteria.preprocessing.scalers import VectorScaler, SumScaler
from skcriteria.preprocessing.weighters import (
    EntropyWeighter,
)
from skcriteria.pipelines import mkpipe
from skcriteria.agg.topsis import TOPSIS

from skcriteria.ranksrev import RankInvariantChecker, RankTransitivityChecker


ws7 = skc.datasets.load_van2021evaluation(windows_size=7)
ws15 = skc.datasets.load_van2021evaluation(windows_size=15)

dm = mkpipe(
    NegateMinimize(),
    EntropyWeighter(),  
    VectorScaler(target="matrix"),        
    TOPSIS())

    
inv_chk = RankInvariantChecker(dm)

inv_chk.evaluate(ws15)
