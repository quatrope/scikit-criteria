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


# from skcriteria import datasets
# dm = datasets.load_van2021evaluation()
# dm
# alternatives = list(dm.alternatives)

# import networkx as nx

# import matplotlib.pyplot as plt
# import networkx as nx

# def plot(G):
#     # 3. Configure the drawing with visible labels
#     nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", node_size=700)
#     # 4. Display the window containing the rendering
#     plt.show()


# nodos = ['BNB', 'ADA', 'BTC', 'DOGE', 'ETH', 'LINK', 'LTC', 'XLM', 'XRP']

# aristas = [
#     ('ADA', 'DOGE'),
#     ('BNB', 'ADA'), ('BNB', 'DOGE'), ('BNB', 'LINK'), ('BNB', 'LTC'), ('BNB', 'XLM'), ('BNB', 'XRP'),
#     ('BTC', 'ADA'), ('BTC', 'BNB'), ('BTC', 'DOGE'), ('BTC', 'ETH'), ('BTC', 'LINK'), ('BTC', 'LTC'), ('BTC', 'XLM'), ('BTC', 'XRP'),
#     ('ETH', 'ADA'), ('ETH', 'BNB'), ('ETH', 'DOGE'), ('ETH', 'LINK'), ('ETH', 'XLM'), ('ETH', 'XRP'),
#     ('LINK', 'ADA'), ('LINK', 'DOGE'), ('LINK', 'XLM'), ('LINK', 'XRP'),
#     ('LTC', 'ADA'), ('LTC', 'DOGE'), ('LTC', 'ETH'), ('LTC', 'LINK'), ('LTC', 'XLM'), ('LTC', 'XRP'),
#     ('XLM', 'ADA'), ('XLM', 'DOGE'),
#     ('XRP', 'ADA'), ('XRP', 'DOGE'), ('XRP', 'XLM'),
# ]

# g = nx.DiGraph()
# g.add_nodes_from(nodos)
# g.add_edges_from(aristas)
# G = g


# def as_condensed_dag(graph):
#     condensed = nx.condensation(graph)
#     dag = nx.transitive_reduction(condensed)
#     dag.add_nodes_from(condensed.nodes(data=True))
#     members, labels = {}, {}
#     for node, data in dag.nodes(data=True):

#         node_members = data["members"]
#         node_name = "+\n".join(node_members)

#         members[node_name] = node_members
#         labels[node] = node_name

#     nx.relabel_nodes(dag, labels, copy=False)
#     return dag, members

# import numpy as np

# def ranking_from_generations(alternatives, dag, members):
#     """Generate a ranking based on topological generations.

#     Creates a single ranking where alternatives in the same topological
#     generation (incomparable elements) share the same rank. This provides
#     a compact representation when ties are acceptable.

#     Meant to be used with the condensed DAG from :func:`as_condensed_reduced_dag`,
#     where a node may represent several alternatives tied together by a
#     dominance cycle.

#     Parameters
#     ----------
#     alternatives : array-like
#         Array of alternative names/identifiers in their original order.
#         This defines the order in which ranks are returned in the ranking.
#     dag : networkx.DiGraph
#         A directed acyclic graph representing preference relations, as
#         returned by :func:`as_condensed_reduced_dag`.
#     members : dict
#         Maps each node of ``dag`` to the set of alternatives it
#         represents, as returned by :func:`as_condensed_reduced_dag`.

#     Returns
#     -------
#     np.ndarray
#         A 1-indexed NumPy array where the i-th element is the rank of the
#         i-th alternative. Alternatives in the same generation (i.e. in
#         the same dominance cycle) share the same rank. Lower ranks
#         indicate better alternatives.

#     """
#     # Map each alternative to its generation number (1-indexed)
#     alt_to_rank = {}
#     for rank, generation in enumerate(
#         nx.topological_generations(dag), start=1
#     ):
#         gen_members = members[generation[0]]
#         for alt in gen_members:
#             alt_to_rank[alt] = rank

#     # Build rank array in original alternative order
#     ranking = np.array(
#         [alt_to_rank[alt] for alt in alternatives],
#         dtype=int,
#     )
#     return ranking

# import itertools as it

# def generate_rankings_from_x(alternatives, dag, members, *, max_ranks=None):
#     """[CLAUDE COMPLETA]
#     """
#     # [CLAUDE COMPLETA]
#     all_permutations = []
#     for generation in nx.topological_generations(dag):

#         # [CLAUDE COMPLETA]
#         gen_members = members[generation[0]]

#         # [CLAUDE COMPLETA]
#         generation_permutations = it.permutations(gen_members)
#         all_permutations.append(generation_permutations)

#     # [CLAUDE COMPLETA]
#     import ipdb; ipdb.set_trace()
#     generated_rankins = 0
#     for permutation in it.product(*all_permutations):
#         if max_ranks is not None and generated_rankins >= max_ranks:
#             break

#         # [CLAUDE no me gusta el nombre plain_permutation]
#         plain_permutation = it.chain(*permutation)

#         alt_to_rank = {alternative: rank for rank, alternative in enumerate(plain_permutation, start=1)}
#         import ipdb; ipdb.set_trace()

#         ranking = np.array(
#             [alt_to_rank[alt] for alt in alternatives],
#             dtype=int,
#         )

#         yield ranking
#         generated_rankins += 1

# dag, mem = as_condensed_dag(G)


# #plot(cg)

# len(list(generate_rankings_from_x(alternatives, dag, mem)))
