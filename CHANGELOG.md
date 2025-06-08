# Whats new?

<!-- BODY -->

## Version 0.9

- **New**: The `has_loops`method of the `DecisionMatrix` class now uses the networkx library to compute the dominance graph and check if it is a DAG.
- **New**: Added the `to_latex` method to `DecisionMatrix` to generate a LaTeX table representation of the decision matrix.
- **Changed Behavior**: Corrected the functionality of the PushNegatives transformation. The transformer now appropriately manages negative values by identifying the lowest negative value in the matrix and adding its absolute value to all matrix elements.

- **Changed Behaviour**: The `copy()` method with `kwargs` in `DecisionMatrix` and all `SKCMethodABC` derived classes is now deprecated. Instead, use the new `replace()` method which provides a clearer and more maintainable way to create copies with modified parameters:

    ```python
    # Old way (deprecated)
    dm_copy = decision_matrix.copy(weights=[0.5, 0.5])
    method_copy = method.copy(parameter=new_value)

    # New way
    dm_copy = decision_matrix.replace(weights=[0.5, 0.5])
    method_copy = method.replace(parameter=new_value)
    ```

  The `copy()` method will continue to work for creating exact copies, but passing
  parameters through `kwargs` will raise a deprecation warning.

- The `utils.deprecate` module has been streamlined for better usability. The warning system now uses a simpler implementation that emits warnings only once per call location. The raising functionality has been moved to module level configuration.

- ***Dependencies:** Added requirements for NumPy 2.0, Network-X 3.4 and scikit-learn 1.6


## Version 0.8.7

- **New** Added functionality for user extension of scikit-criteria with
  decorators for creating aggregation and transformation models using
  functions.

    ```python
    >>> from skcriteria.extend import mkagg, mktransformer
    >>>
    >>> @mkagg
    >>> def MyAgg(**kwargs):
    >>>     # Implementation of the aggregation function
    >>>
    >>> @mkagg(foo=1)
    >>> def MyAggWithHyperparam(**kwargs):
    >>>     # Implementation of the aggregation function with
    >>>     # hyperparameter 'foo'
    >>>
    >>> @mktransformer
    >>> def MyTransformer(**kwargs):
    >>>     # Implementation of the transformation function
    >>>
    >>> @mktransformer(bar=2)
    >>> def MyTransformerWithHyperparam(**kwargs):
    >>>     # Implementation of the transformation function with
    >>>     # hyperparameter 'bar'
    ```

  These decorators enable the creation of aggregation and transformation
  classes based on provided functions, allowing users to
  define decision-making models with less flexibility than traditional
  inheritance-based models.

  For more information check the tutorial [Extending Aggregation and Transformation Functions](https://scikit-criteria.quatrope.org/en/latest/tutorial/extend.html)

- **New Module:** Introduced the `skcriteria.testing` module, exposing utility functions for for comparing objects created in Scikit-Criteria in a testing environment. These functions facilitate the comparison of instances of the `DecisionMatrix`, `ResultABC`, and `RanksComparator` classes.

  The assertion functions utilize pandas and numpy testing utilities for comparing matrices, series, and other attributes.

  Check the [Reference](https://scikit-criteria.quatrope.org/en/latest/api/testing.html) for more information.

- **New** The API of the agg, pipeline, preprocessing, and extend modules has
  been cleaned up to prevent autocompletion with imports from other modules.
  The imported modules are still present, but they are excluded when attempting
  to autocomplete. This functionality is achieved thanks to the context manager
  `skcriteria.utils.cmanagers.hidden()`.

- **New** All methods (agg and transformers) has a new `get_method_name`
  instance method.

- **Drop** Drop support for Python 3.8

---

## Version 0.8.6

- **New** Rank reversal 1 implementhed in the `RankInvariantChecker` class

    ```python

    >>> import skcriteria as skc
    >>> from skcriteria.cmp import RankInvariantChecker
    >>> from skcriteria.agg.similarity import TOPSIS

    >>> dm = skc.datasets.load_van2021evaluation()
    >>> rrt1 = RankInvariantChecker(TOPSIS())
    >>> rrt1.evaluate(dm)
    <RanksComparator [ranks=['Original', 'M.ETH', 'M.LTC', 'M.XLM', 'M.BNB', 'M.ADA', 'M.LINK', 'M.XRP', 'M.DOGE']]>
    ```

- **New** The module `skcriteria.madm` was deprecated in favor
  of `skcriteria.agg`
- Add support for Python 3.11.
- Removed Python 3.7. Google collab now work with 3.8.
- Updated Scikit-Learn to 1.3.x.
- Now all cached methods and properties are stored inside the instance.
  Previously this was stored inside the class generating a memoryleak.

---

## Version 0.8.3

- Fixed a bug detected on the EntropyWeighted, Now works as the literature
  specifies

---

## Version 0.8.2

- We bring back Python 3.7 because is the version used in google.colab.
- Bugfixes in `plot.frontier` and `dominance.eq`.

---

## Version 0.8

- **New** The `skcriteria.cmp` package utilities to compare rankings.

- **New** The new package `skcriteria.datasets` include two datasets (one a
  toy and one real) to quickly start your experiments.

- **New** DecisionMatrix now can be sliced with a syntax similar of the
  pandas.DataFrame.
  - `dm["c0"]` cut the *c0* criteria.
  - `dm[["c0", "c2"]` cut the criteria *c0* and *c2*.
  - `dm.loc["a0"]` cut the alternative *a0*.
  - `dm.loc[["a0", "a1"]]` cut the alternatives *a0* and *a1*.
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

  - `Objective.construct_from_alias()` `->` `Objective.from_alias()` (*classmethod*)
  - `Objective.to_string()` `->` `Objective.to_symbol()`

  The deprecated methods will be removed in version *1.0*.

- Add a dominance plot `DecisionMatrix.plot.dominance()`.

- `WeightedSumModel` raises a `ValueError` when some value *< 0*.

- Moved internal modules
  - `skcriteria.core.methods.SKCTransformerABC` `->`
    `skcriteria.preprocessing.SKCTransformerABC`
  - `skcriteria.core.methods.SKCMatrixAndWeightTransformerABC` `->`
    `skcriteria.preprocessing.SKCMatrixAndWeightTransformerABC`

---

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
  - `RankingResult.untied_rank_` to get a ranking with no repeated values.
      repeated values.
- `KernelResult` now implements several new properties:

  - `kernel_alternatives_` to know which alternatives are in the kernel.
  - `kernel_size_` to know the number of alternatives in the kernel.
  - `kernel_where_` was replaced by `kernelwhere_` to standardize the api.

---

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

---

## Version 0.5

In this version scikit-criteria was rewritten from scratch. Among other things:

- The model implementation API was simplified.
- The `Data` object was removed in favor of `DecisionMatrix` which implements many more useful features for MCDA.
- Plots were completely re-implemented using [Seaborn](http://seaborn.pydata.org/).
- Coverage was increased to 100%.
- Pipelines concept was added (Thanks to [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)).
- New documentation. The quick start is totally rewritten!

**Full Changelog**: https://github.com/quatrope/scikit-criteria/commits/0.5

---

## Version 0.2

First OO stable version.

---

## Version 0.1

Only functions.
