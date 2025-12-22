# AGENT.md

## Important Rules

**NEVER perform `git push` or `git commit` without explicit permission from the user.**

## Project Overview

**scikit-criteria**: Python library for Multi-Criteria Decision Analysis (MCDA) providing algorithms and methods for solving decision-making problems with multiple conflicting criteria.

```
skcriteria/
├── __init__.py
├── core/                # Core data structures (DecisionMatrix, Objective, mkdm)
│   ├── data.py          # DecisionMatrix and factory functions
│   ├── objectives.py    # Objective definitions (minimize, maximize)
│   ├── methods.py       # Base classes for MCDA methods (SKCMethodABC)
│   └── plot.py          # Visualization tools (DecisionMatrixPlotter)
├── agg/                 # Decision-making aggregation methods (MABAC, etc.)
│   ├── _agg_base.py     # Base classes (SKCDecisionMakerABC, RankResult, KernelResult)
│   └── mabac.py         # MABAC and other aggregation methods
├── preprocessing/       # Data transformation and preprocessing
│   ├── _preprocessing_base.py  # Base transformer (SKCTransformerABC)
│   ├── scalers.py       # Normalization and scaling transformers
│   ├── weighters.py     # Weight calculation methods
│   ├── filters.py       # Criteria/alternative filtering
│   └── impute.py        # Missing value imputation
├── pipelines/           # Pipeline composition for chaining transformers and methods
├── ranksrev/            # Rank reversal detection and mitigation
├── cmp/                 # Result comparison utilities
├── datasets/            # Sample datasets for examples and testing
├── io/                  # Import/export operations (read_dmsy, to_dmsy)
├── utils/               # Utility functions and helpers
│   ├── ondemand_import.py  # Lazy loading of submodules
│   ├── deprecate.py     # Deprecation warning utilities
│   └── dag_rank.py      # DAG-based ranking (FAS algorithm)
├── extend.py            # Decorators for creating custom models from functions
├── tiebreaker.py        # Tie-breaking strategies for ranking
└── testing.py           # Testing utilities for MCDA methods
```

## Key Patterns

- **Lazy Module Loading**: Submodules loaded on-demand via `__getattr__` using `ondemand_import`
- **Immutable Decision Matrices**: DecisionMatrix objects are frozen dataclasses with copy-on-modify semantics
- **ABC-Based Extension**: All methods inherit from `SKCMethodABC`, transformers from `SKCTransformerABC`
- **Result Hierarchy**: Methods return `RankResult` or `KernelResult` subclasses of `ResultABC`
- **Pipeline Composition**: Transformers and decision makers chainable via `SKCPipeline`/`mkpipe`
- **Decorator-Based Models**: `@agg_method` and `@transformer` decorators create models from functions
- **Backward Compatibility**: Deprecated modules (`madm`, `pipeline`) preserved with warnings


## Code Style

- pep-8
- Code formater: 'black -l 79'
- Max 79 columns per line.
- NumPy-style docstrings (NO Examples section)
- Module organization: docstring → imports → constants → private helpers → classes → public functions
- Import order (PEP 8):
  1. Standard library imports (alphabetically)
  2. Third-party library imports (alphabetically)
  3. Local/application imports (alphabetically)
  4. Each group separated by a blank line
