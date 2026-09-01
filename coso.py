"""Minimal repro to debug CriteriaOneAtATimeChecker._post_process_rank_comparator."""

import skcriteria as skc
from skcriteria.agg import simple
from skcriteria.importance import CriteriaOneAtATimeChecker

dm = skc.mkdm(
    matrix=[[7, 2, 9], [4, 8, 2], [3, 5, 6], [9, 1, 4]],
    objectives=[max, max, max],
    weights=[0.5, 0.3, 0.2],
    criteria=["C0", "C1", "C2"],
)

checker = CriteriaOneAtATimeChecker(simple.WeightedSumModel(), delta=0.5)

rank_cmp = checker.evaluate(dm)

print("ranks:", [name for name, _ in rank_cmp.ranks])
print()
print("importance:")
print(rank_cmp.extra_["importance"])
