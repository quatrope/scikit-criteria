.. FILE AUTO GENERATED !! 

Version 0.8
-----------


* Add a dominance plot ``DecisionMatrix.plot.dominance()``.

Version 0.7
-----------


* **New method**\ : ``ELECTRE2``.
* **New preprocessing strategy:** A new way to transform  from minimization to
  maximization criteria: ``NegateMinimize()`` which  reverses the sign of the
  values of the criteria to be minimized (useful for not breaking distance
  relations in methods like *TOPSIS*\ ). Additionally the previous we rename the
  ``MinimizeToMaximize()`` transformer to ``InvertMinimize()``.
* 
  Now the ``RankingResult``\ , support repeated/tied rankings and some methods were
  implemented to deal with these cases.


  * ``RankingResult.has_ties_`` to see if there are tied values.
  * ``RankingResult.ties_`` to see how often values are repeated.
  * ``RankingResult.untided_rank_`` to get a ranking with no repeated values.
      repeated values.

* 
  ``KernelResult`` now implements several new properties:


  * ``kernel_alternatives_`` to know which alternatives are in the kernel.
  * ``kernel_size_`` to know the number of alternatives in the kernel.
  * ``kernel_where_`` was replaced by ``kernelwhere_`` to standardize the api.

Version 0.6
-----------


* Support for Python 3.10.
* All the objects of the project are now immutable by design, and can only
  be mutated troughs the ``object.copy()`` method.
* Dominance analysis tools (\ ``DecisionMatrix.dominance``\ ).
* The method ``DecisionMatrix.describe()`` was deprecated and will be removed
  in version *1.0*.
* New statistics functionalities ``DecisionMatrix.stats`` accessor.
* 
  The accessors are now cached in the ``DecisionMatrix``.

* 
  Tutorial for dominance and satisfaction analysis.

* 
  TOPSIS now support hyper-parameters to select different metrics.

* Generalize the idea of accessors in scikit-criteria througth a common
  framework (\ ``skcriteria.utils.accabc`` module).
* New deprecation mechanism through the
* ``skcriteria.utils.decorators.deprecated`` decorator.

Version 0.5
-----------

In this version scikit-criteria was rewritten from scratch. Among other things:


* The model implementation API was simplified.
* The ``Data`` object was removed in favor of ``DecisionMatrix`` which implements many more useful features for MCDA.
* Plots were completely re-implemented using `Seaborn <http://seaborn.pydata.org/>`_.
* Coverage was increased to 100%.
* Pipelines concept was added (Thanks to `Scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_\ ).
* New documentation. The quick start is totally rewritten!

**Full Changelog**\ : https://github.com/quatrope/scikit-criteria/commits/0.5

Version 0.2
-----------

First OO stable version.

Version 0.1
-----------

Only functions.
