import skcriteria as skc
from skcriteria.agg.topsis import TOPSIS
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler
from skcriteria.preprocessing.weighters import EntropyWeighter
from skcriteria.ranksrev.rank_transitivity_check import RankTransitivityChecker

pipe = mkpipe(
    InvertMinimize(),
    EntropyWeighter(),
    SumScaler(target="matrix"),
    TOPSIS(),
)

checker = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

dm = skc.datasets.load_van2021evaluation()

result = checker.evaluate(dm)

print(result)
