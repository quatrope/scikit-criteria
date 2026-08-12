# Planificación de tutoriales — scikit-criteria

Este documento tiene tres partes:

1. **Features disponibles** en la librería (inventario por módulo).
2. **Tutoriales que ya existen** en `docs/source/tutorial`.
3. **Planificaciones propuestas** (≤ 5 partes cada una) para los features que
   todavía no tienen tutorial dedicado, o que solo están cubiertos
   parcialmente.

La idea es que marques qué planificación atacamos primero y en qué orden.

---

## 1. Features disponibles

### 1.1 Core (`skcriteria.core`)
- `DecisionMatrix` / `mkdm`: estructura central (alternativas × criterios,
  objetivos, pesos).
- `DecisionMatrix.plot`: accessor de gráficos (heatmap, radar, box, violin,
  scatter, bar, etc. sobre la matriz).
- `DecisionMatrix.stats`: accessor de estadística descriptiva.
- `DecisionMatrix.dominance`: accessor de análisis de dominancia
  (`dominance`, `satisfaction`, `frontier`, etc.).
- `Objective`: enum de sentido de criterios (MAX/MIN).

### 1.2 Datasets (`skcriteria.datasets`)
- `load_simple_stock_selection()`
- `load_van2021evaluation()`

### 1.3 Preprocessing (`skcriteria.preprocessing`)
- **Scalers** (`scalers.py`): `StandarScaler`, `MinMaxScaler`,
  `MaxAbsScaler`, `MaxScaler`, `VectorScaler`, `SumScaler`,
  `CenitDistanceMatrixScaler`.
- **Weighters** (`weighters.py`): `EqualWeighter`, `StdWeighter`,
  `EntropyWeighter`, `CRITIC`, `MEREC`, `GiniWeighter`, `RANCOM`.
- **Filters** (`filters.py`): `Filter`, `FilterGT/GE/LT/LE/EQ/NE`,
  `FilterIn`, `FilterNotIn`, `FilterNonDominated`.
- **Invert objectives** (`invert_objectives.py`): `NegateMinimize`,
  `InvertMinimize`, `MinimizeToMaximize`, `MinMaxInverter`,
  `BenefitCostInverter`.
- **Impute** (`impute.py`): `SimpleImputer`, `IterativeImputer`,
  `KNNImputer`.
- **Increment** (`increment.py`): `AddValueToZero`.
- **Push negatives** (`push_negatives.py`): `PushNegatives`.
- **Distance** (`distance.py`): `CenitDistance`.

### 1.4 Métodos de agregación / ranking (`skcriteria.agg`)
Cada uno es un `SKCDecisionMakerABC` completo (24 métodos):

`WeightedSumModel`, `WeightedProductModel`, `TOPSIS`, `ELECTRE1`,
`ELECTRE2`, `EDAS`, `CoCoSo`, `ARAS`, `COPRAS`, `CODAS`, `MARCOS`,
`RatioMOORA`, `ReferencePointMOORA`, `FullMultiplicativeForm`,
`MultiMOORA`, `ERVD`, `MAIRCA`, `MABAC`, `SPOTIS`, `OCRA`, `RIM`,
`SIMUS`, `RAM`, `PROBID`/`SimplifiedPROBID`, `VIKOR`, `WASPAS`.

### 1.5 Comparación de rankings (`skcriteria.cmp`)
- `RanksComparator` / `mkrank_cmp`: compara varios rankings lado a lado.
- `RanksComparatorPlotter`: gráficos comparativos (flow, box, corr, etc.).

### 1.6 Rank reversal (`skcriteria.ranksrev`)
- `RankInvariantChecker` (Wang RRT1: reemplazo por alternativa peor).
- `RankClonesChecker` (Belton & Gear: clon de alternativa no óptima).
- `RankTransitivityChecker` (Wang TC2/TC3: transitividad y reconstrucción).

### 1.7 Desempate (`skcriteria.tiebreaker`)
- `FallbackTieBreaker`: resuelve empates de un método con un método de
  respaldo.

### 1.8 Pipelines (`skcriteria.pipelines` / `skcriteria.pipeline`)
- `SKCPipeline` / `mkpipe`: encadenar preprocessing + método de agregación.
- `SKCCombinatorialPipeline` / `mkcombinatorial`: correr todas las
  combinaciones posibles de una lista de pasos alternativos.

### 1.9 Extensión (`skcriteria.extend`)
- `mkagg`: convertir una función en un método de agregación custom.
- `mktransformer`: convertir una función en un transformer custom.

### 1.10 Entrada/salida (`skcriteria.io`)
- `read_dmsy` / `to_dmsy`: serialización de `DecisionMatrix` a/desde YAML
  (formato DMSY).

### 1.11 Utilidades (`skcriteria.utils`)
- `rank.py`: `rank_values`, `is_rank`, `dominance`.
- `lp.py`: wrappers de programación lineal (`Float`, `Int`, `Bool`,
  `Minimize`, `Maximize`) usados internamente por SIMUS/SPOTIS/etc.
- `dag_rank.py`: conversión de grafos de dominancia a DAG y ranking por
  generaciones (usado por `RankTransitivityChecker`).

---

## 2. Tutoriales que ya existen (`docs/source/tutorial`)

| Archivo | Título | Contenido actual |
|---|---|---|
| `quickstart.ipynb` | Quick Start | Manipular `DecisionMatrix`, plotting básico, transformación de datos, alimentar TOPSIS y ELECTRE |
| `sufdom.ipynb` | Dominance and satisfaction analysis (AKA filters) | Satisfacción, dominancia, `FilterNonDominated`, experimento completo |
| `rankcmp.ipynb` | Rankings comparison | `RanksComparator`, utilidades y gráficos de comparación |
| `rankrev.ipynb` | Rank reversals | Test criterion 1 (invariancia), criterios 2 y 3 (transitividad/reconstrucción), ejemplos que pasan/fallan |
| `scale_weight.ipynb` | Scaling and weighting criteria | **✅ Hecho** — scalers (MinMax/Standar/Vector/MaxAbs), weighters estadísticos (Equal/Std/Entropy/Gini) y de correlación/expertos (CRITIC/MEREC/RANCOM), combinados en pipelines y comparados con `RanksComparator` |
| `extend.ipynb` | Extending Aggregation and Transformation Functions | Nuevo modelo de agregación, hiperparámetros, nuevo transformer, consideraciones de `dtype` |

**Cobertura actual:** `DecisionMatrix` básico, TOPSIS, ELECTRE (mención),
filtros de dominancia, comparación de rankings, rank reversal (los 3
checkers), y extensión de la librería.

**Sin tutorial dedicado:** el resto de los 24 métodos de agregación,
scalers/weighters/imputers en profundidad, pipelines, combinatorial
pipelines, tiebreaker, I/O (dmsy), y `stats`/`plot` accessors en detalle.

---

## 3. Planificaciones propuestas (≤ 5 partes)

### A. Preprocessing: Scalers y Weighters — ✅ Hecho (`tutorial/scale_weight.ipynb`)
1. Por qué normalizar/pesar: problema con escalas y unidades distintas.
2. Scalers disponibles: `MinMaxScaler`, `StandarScaler`, `VectorScaler`,
   `SumScaler`, `MaxAbsScaler`/`MaxScaler` — cuándo usar cada uno.
3. Weighters "estadísticos": `EqualWeighter`, `StdWeighter`,
   `EntropyWeighter`, `GiniWeighter`.
4. Weighters "de correlación/expertos": `CRITIC`/`Critic`, `MEREC`,
   `RANCOM`.
5. Combinar scaler + weighter en un mismo flujo y comparar el impacto en
   un ranking (TOPSIS) según la combinación elegida.

### B. Preprocessing: Filtros y datos faltantes
1. `Filter` genérico y filtros aritméticos (`FilterGT/GE/LT/LE/EQ/NE`).
2. Filtros por conjunto (`FilterIn`, `FilterNotIn`) y de dominancia
   (`FilterNonDominated` — referenciar `sufdom.ipynb` para no duplicar).
3. Datos faltantes: `SimpleImputer`.
4. Imputación avanzada: `IterativeImputer`, `KNNImputer` — comparación.
5. Caso combinado: filtrar alternativas inválidas + imputar + rankear.

### C. Preprocessing: Objetivos e invertir criterios
1. El problema de mezclar criterios MAX/MIN en un mismo método.
2. `NegateMinimize` vs `InvertMinimize`/`MinimizeToMaximize`: diferencia
   conceptual (negar vs invertir).
3. `MinMaxInverter` y `BenefitCostInverter`: casos de uso.
4. `AddValueToZero` y `PushNegatives`: por qué algunos métodos (ej. los
   basados en ratios) no toleran ceros o negativos.
5. Ejercicio integrador: preparar una matriz "sucia" (mixta, con ceros y
   negativos) para que sea compatible con un método sensible como MOORA.

### D. Métodos de agregación clásicos (más allá de TOPSIS/ELECTRE)
1. Repaso rápido de la familia "suma/producto ponderado":
   `WeightedSumModel`, `WeightedProductModel`, `WASPAS`.
2. Familia MOORA: `RatioMOORA`, `ReferencePointMOORA`,
   `FullMultiplicativeForm`, `MultiMOORA` (agregación de sub-métodos).
3. Familia de distancia a soluciones ideales: `VIKOR`, `CODAS`, `MABAC`,
   `MAIRCA`.
4. Familia basada en ranking agregado/comparación: `ARAS`, `COPRAS`,
   `CoCoSo`, `EDAS`, `MARCOS`, `OCRA`, `RAM`.
5. Comparar los resultados de todos con `RanksComparator` sobre el mismo
   dataset (`load_simple_stock_selection`) y discutir por qué difieren.

### E. ELECTRE en profundidad
1. Repaso del quickstart: qué hace `ELECTRE1` (concordancia/discordancia,
   relación de outranking).
2. `ELECTRE2`: diferencias con ELECTRE1 (umbrales fuerte/débil, dos
   relaciones de outranking).
3. Cómo interpretar la salida (no siempre es un ranking total).
4. Sensibilidad a los umbrales: variar parámetros y ver el impacto.
5. Cuándo elegir ELECTRE sobre TOPSIS/WSM (criterios no compensatorios).

### F. Métodos con programación lineal: SIMUS y SPOTIS
1. Motivación: métodos que resuelven un problema de optimización en vez
   de una fórmula cerrada.
2. `SIMUS`: formulación, escenarios, y cómo se apoya en
   `skcriteria.utils.lp`.
3. Ejemplo completo con `SIMUS` y lectura de resultados
   (`stages`, `stages_results`).
4. `SPOTIS`: idea del punto de referencia acotado por límites.
5. Comparar `SIMUS`/`SPOTIS` contra TOPSIS en el mismo dataset.

### G. RIM, ERVD y PROBID (métodos "de referencia")
1. Idea común: comparar alternativas contra un punto/región de referencia
   en vez de solo el ideal.
2. `RIM` (Reference Ideal Method): región ideal por criterio.
3. `ERVD`: función de valor con referencia (basada en teoría de
   prospectos).
4. `PROBID` / `SimplifiedPROBID`: distancia probabilística a un conjunto
   de soluciones ideales.
5. Ejercicio: elegir el método correcto según qué tipo de "referencia"
   tiene sentido en el dominio del problema.

### H. Pipelines
1. Por qué encadenar pasos: repetir preprocessing + método a mano es
   propenso a errores.
2. `mkpipe`/`SKCPipeline`: armar un pipeline scaler → weighter → método.
3. Inspeccionar un pipeline (`steps`, acceso a cada paso, resultados
   intermedios).
4. `mkcombinatorial`/`SKCCombinatorialPipeline`: correr todas las
   combinaciones de una lista de pasos alternativos (ej. 3 scalers × 2
   weighters × 1 método).
5. Usar `RanksComparator` sobre los resultados del combinatorial pipeline
   para elegir la mejor combinación.

### I. Desempates con `FallbackTieBreaker`
1. El problema: un método produce empates (ranking no estricto).
2. `FallbackTieBreaker`: método principal + método de respaldo.
3. Ejemplo con un método propenso a empates (ej. `ELECTRE1`) + TOPSIS
   como fallback.
4. `TieUnresolvedWarning`: qué pasa si ni el fallback logra desempatar.
5. Buenas prácticas: cuándo desempatar automáticamente vs. reportar el
   empate.

### J. Entrada/salida con formato DMSY
1. Por qué serializar una `DecisionMatrix` (reproducibilidad, compartir
   casos de estudio).
2. `to_dmsy`: guardar una matriz a YAML.
3. `read_dmsy`: leer y reconstruir la matriz (verificar que
   alternativas/objetivos/pesos se preserven).
4. Inspeccionar el archivo YAML generado a mano (formato legible).
5. Caso de uso: compartir un dataset propio como archivo `.dmsy` para que
   otro lo reproduzca.

### K. `stats` y `plot` accessors en profundidad
1. Repaso del quickstart: qué gráficos ya se vieron (heatmap básico).
2. `DecisionMatrix.stats`: estadística descriptiva por criterio/alternativa.
3. Tour de `DecisionMatrix.plot`: box, violin, radar, scatter, bar —
   cuándo usar cada uno para explorar el problema antes de decidir.
4. Combinar `plot` + `dominance` para visualizar el frente no dominado.
5. Checklist de "exploración de datos" antes de aplicar un método MCDA.

### L. Crear métodos y transformers custom con `extend`
*(Ya existe `extend.ipynb`; esta planificación es una posible versión
ampliada/alternativa si se quiere profundizar más en `mkagg`/
`mktransformer` con casos reales.)*
1. Repaso: `mkagg` para envolver una función simple como método de
   agregación.
2. Manejo de hiperparámetros custom y validación de nombres
   (`NonStandardNameWarning`).
3. `mktransformer` para un preprocessing custom (ej. una normalización
   propia no incluida en la librería).
4. Registrar el nuevo método en un `pipeline` junto con los provistos por
   la librería.
5. Buenas prácticas y limitaciones (qué no se puede extender así).

---

## Cómo seguimos

Decime con cuál planificación (A–L) arrancamos y en qué orden querés las
demás; puedo ajustar el número de partes, fusionar temas, o descartar los
que no interesen antes de empezar a escribir el notebook.
