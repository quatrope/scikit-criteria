## Changelog of Scikit-Criteria

## 0.6

- TOPSIS now support hyper-parameters to select different metrics.

## 0.5

In this version scikit-criteria was rewritten from scratch. Among other things:

- The model implementation API was simplified.
- The `Data` object was removed in favor of `DecisionMatrix` which implements many more useful features for MCDA.
- Plots were completely re-implemented using [Seaborn](http://seaborn.pydata.org/).
- Coverage was increased to 100%.
- Pipelines concept was added (Thanks to [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)).
- New documentation. The quick start is totally rewritten!

**Full Changelog**: https://github.com/quatrope/scikit-criteria/commits/0.5

## 0.2

First OO stable version.

## 0.1

Only functions.