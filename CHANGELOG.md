# Changelog of Scikit-Criteria

<!-- BODY -->

## Version 0.8

- **New** The `skcriteria.cmp` package utilities to compare rankings.

- **New** The new package `skcriteria.datasets` include two datasets (one a
  toy and one real) to quickly start your experiments.

- **New** DecisionMatrix now can be sliced with a syntax similar of the
  pandas.DataFrame.
  - `dm["c0"]` cut the $c0$ criteria.
  - `dm[["c0", "c2"]` cut the criteria $c0$ and $c2$.
  - `dm.loc["a0"]` cut the alternative $a0$.
  - `dm.loc[["a0", "a1"]]` cut the alternatives $a0$ and $a1$.
  - `dm.iloc[0:3]` cuts from the first to the third alternative.

- **New** imputation methods for replacing missing data with substituted
  values. These methods are in the module `skcriteria.preprocessing.impute`.

- **New** results object now has a `to_series` method.

- **Changed Behaviour**: The ranks and kernels `equals` are now called
  `values_equals`. The new `aequals` support tolerances to compare
  numpy arrays internally stored in `extra_`, and the `equals` method is
  equivalent to `aequals(rtol=0, atol=0)`.

- We detected a bad behavior in ELECTRE2, so we decided to launch a `FutureWarning` when the
  class is instantiated. In the version after 0.8, a new implementation of ELECTRE2 will be
  provided.

- Multiple `__repr__` was improved to folow the
  [Python recomendation](https://docs.python.org/3/library/functions.html#repr)

- `Critic` weighter was renamed to `CRITIC` (all capitals) to be consistent
  with the literature. The old class is still there but is deprecated.

- All the functions and classes of `skcriteria.preprocessing.distance` was
  moved to `skcriteria.preprocessing.scalers`.

- The `StdWeighter` now uses the **sample** standar-deviation.
  From the numerical point of view, this does not generate any change,
  since the deviations are scaled by the sum. Computationally speaking there
  may be some difference from the ~5th decimal digit onwards.

- Two method of the `Objective` enum was deprecated and replaced:

  - `Objective.construct_from_alias()` `->` `Objective.from_alias()` (classmethod)
  - `Objective.to_string()` `->` `Objective.to_symbol()'`

  The deprecated methods will be removed in version *1.0*.

- Add a dominance plot `DecisionMatrix.plot.dominance()`.

- `WeightedSumModel` raises a `ValueError` when some value $< 0$.

- Moved internal modules
  - `skcriteria.core.methods.SKCTransformerABC` `->`
    `skcriteria.preprocessing.SKCTransformerABC`
  - `skcriteria.core.methods.SKCMatrixAndWeightTransformerABC` `->`
    `skcriteria.preprocessing.SKCMatrixAndWeightTransformerABC`

## Version 0.7

- **New method**: `ELECTRE2`.
- **New preprocessing strategy:** A new way to transform  from minimization to
  maximization criteria: `NegateMinimize()` which  reverses the sign of the
  values of the criteria to be minimized (useful for not breaking distance
  relations in methods like *TOPSIS*). Additionally the previous we rename the
  `MinimizeToMaximize()` transformer to `InvertMinimize()`.
- Now the `RankingResult`, support repeated/tied rankings and some methods were
  implemented to deal with these cases.

  - `RankingResult.has_ties_` to see if there are tied values.
  - `RankingResult.ties_` to see how often values are repeated.
  - `RankingResult.untided_rank_` to get a ranking with no repeated values.
      repeated values.
- `KernelResult` now implements several new properties:

  - `kernel_alternatives_` to know which alternatives are in the kernel.
  - `kernel_size_` to know the number of alternatives in the kernel.
  - `kernel_where_` was replaced by `kernelwhere_` to standardize the api.


## Version 0.6

- Support for Python 3.10.
- All the objects of the project are now immutable by design, and can only
  be mutated troughs the `object.copy()` method.
- Dominance analysis tools (`DecisionMatrix.dominance`).
- The method `DecisionMatrix.describe()` was deprecated and will be removed
  in version *1.0*.
- New statistics functionalities `DecisionMatrix.stats` accessor.
- The accessors are now cached in the `DecisionMatrix`.

- Tutorial for dominance and satisfaction analysis.

- TOPSIS now support hyper-parameters to select different metrics.
- Generalize the idea of accessors in scikit-criteria througth a common
  framework (`skcriteria.utils.accabc` module).
- New deprecation mechanism through the
- `skcriteria.utils.decorators.deprecated` decorator.

## Version 0.5

In this version scikit-criteria was rewritten from scratch. Among other things:

- The model implementation API was simplified.
- The `Data` object was removed in favor of `DecisionMatrix` which implements many more useful features for MCDA.
- Plots were completely re-implemented using [Seaborn](http://seaborn.pydata.org/).
- Coverage was increased to 100%.
- Pipelines concept was added (Thanks to [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)).
- New documentation. The quick start is totally rewritten!

**Full Changelog**: https://github.com/quatrope/scikit-criteria/commits/0.5

## Version 0.2

First OO stable version.

## Version 0.1

Only functions.
