# Bug: `RankInvariantChecker.evaluate()` falla con `ValueError: ... doesn't look like a ranking`

## Síntoma

Al correr `coso.py`:

```python
dm = mkpipe(
    NegateMinimize(),
    EntropyWeighter(),
    VectorScaler(target="matrix"),
    TOPSIS())

inv_chk = RankInvariantChecker(dm)
inv_chk.evaluate(ws15)
```

falla con:

```
RuntimeWarning: invalid value encountered in divide
  return entropy_divergence / np.sum(entropy_divergence)
RuntimeWarning: invalid value encountered in cast
  return stats.rankdata(arr, "dense").astype(np.int64)
...
ValueError: The data [-9223372036854775808 -9223372036854775808 ...] doesn't look like a ranking
```

Reproducible en el 100% de las corridas (probado 5/5 veces con `windows_size=15` y también con `windows_size=7` del dataset `van2021evaluation`). El pipeline llano (`dm.evaluate(ws15)`, sin `RankInvariantChecker`) funciona perfecto — el problema aparece únicamente dentro de `RankInvariantChecker`.

## Cadena causal (de la causa raíz al error final)

1. **`RankInvariantChecker.evaluate()`** muta cada alternativa no-óptima para "empeorarla" un poco y vuelve a correr el pipeline sobre esa matriz mutada (`rank_invariant_check.py:610`, `mrank = dmaker.evaluate(mdm)`).

2. El ruido aplicado en la mutación se calcula en **`_mutate_dm`** (`rank_invariant_check.py:262-323`):
   ```python
   noise = alternative_max_abs_noise.apply(lambda b: random.uniform(0, b))
   noise[dm.maxwhere] *= -1        # negativo si el criterio es MAX
   df.loc[mutate] += noise
   ```
   `b` es la cota superior del ruido uniforme para cada criterio, y viene de **`_maximum_abs_noises`** (`rank_invariant_check.py:203-260`): la diferencia absoluta entre el valor de la alternativa a mutar y el de la "siguiente peor" en el ranking de referencia.

3. **El bug**: `b` no tiene ninguna relación garantizada con la magnitud del valor que se va a mutar. En el dataset `van2021evaluation`, las columnas `xVV` y `sVV` tienen una distribución muy heterogénea (BTC/ETH son 1-2 órdenes de magnitud más grandes que el resto). Cuando dos alternativas consecutivas en el ranking tienen valores muy distintos en esas columnas, `b` termina siendo mucho mayor que el valor real de la alternativa a empeorar, y el ruido sorteado (`uniform(0, b)`) puede **cruzar cero e invertir el signo** del valor.

   Ejemplo concreto con `ws15` (ranking de referencia: `BTC=1, BNB=2, ETH=3, LTC=4, LINK=5, XLM=6, ADA=7, XRP=8, DOGE=9`):

   | mutate | next (peor) | criterio | valor propio | valor del "next" | `b` (cota de ruido) |
   |---|---|---|---|---|---|
   | **BNB** | ETH | `xVV` | `1.316e10` | `2.145e11` | `2.013e11` (¡~15x el valor propio!) |
   | **BNB** | ETH | `sVV` | `2.333e10` | `1.695e11` | `1.462e11` |

   Con `b ≈ 2.01e11` y valor propio `≈ 1.3e10`, cualquier sorteo de ruido mayor a `1.3e10` ya empuja `xVV` a negativo. Esto pasó en la corrida reproducida: `BNB.xVV` mutó de `1.316e10` a `-9.737e10`.

   Otras combinaciones "de riesgo" detectadas en el mismo dataset: `LTC→LINK (xRV)`, `XLM→ADA (xRV, sVV)`, `ADA→XRP (xVV)`, `XRP→DOGE (xRV, sRV)`, y `DOGE` (la última del ranking, que usa la estrategia de mediana — ver más abajo) en `xVV` y `xm`.

4. Ese valor negativo llega a **`EntropyWeighter`** (`skcriteria/preprocessing/weighters.py:214-235`, función `entropy_weights`):
   ```python
   entropy = scipy.stats.entropy(matrix, base=base, axis=0)   # asume datos tipo probabilidad (no-negativos)
   entropy_divergence = 1 - entropy
   return entropy_divergence / np.sum(entropy_divergence)
   ```
   `scipy.stats.entropy` no valida signo. Con un valor negativo mezclado en una columna, la entropía de esa columna da `-inf`, `entropy_divergence` da `inf`, y la división final produce pesos `inf`/`nan`. No hay ninguna validación que capture esto en `entropy_weights` ni en `EntropyWeighter`.

5. Esos pesos rotos arruinan el cálculo de **TOPSIS**, que termina devolviendo scores `nan`.

6. **`rankdata`** (`skcriteria/utils/rank.py:65`) castea esos `nan` a un entero centinela en vez de fallar ahí:
   ```python
   return stats.rankdata(arr, "dense").astype(np.int64)
   ```
   con `nan` esto produce `-9223372036854775808` (el mínimo representable en `int64`), silenciosamente.

7. Recién en **`_agg_base.py:428`**, dentro de `_validate_result`, se detecta que el "ranking" no es válido y se lanza el `ValueError` genérico que se ve en el traceback — varios pasos después del origen real del problema, con un mensaje que no menciona nada de NaN/entropía/mutación.

## ¿Es una regresión de algún commit reciente?

No. Se revisó el historial de los tres archivos involucrados:

- `weighters.py` (`entropy_weights`): sin cambios de lógica, solo *style* commits.
- `rank_invariant_check.py` (`_maximum_abs_noises` / `_mutate_dm`): el diseño de "cota = diferencia con la siguiente peor alternativa, mediana como fallback para la última" está así desde el commit original `49bbfb4` ("semi implemehted rank rev outside comp"), sin cambios funcionales después.
- `rank.py` / `_agg_base.py`: sin cambios recientes en la validación de resultados.

Es un gap de diseño preexistente desde la implementación inicial de `RankInvariantChecker`, expuesto de forma consistente por este dataset en particular debido a su alta dispersión de escala entre alternativas consecutivas del ranking.

## Dónde se usa la estrategia de mediana (`last_diff_strategy`)

1. **Mapeo de nombres** — `rank_invariant_check.py:38-41`:
   ```python
   _LAST_DIFF_STRATEGIES = {"median": np.median, "mean": np.mean}
   ```
2. **Resolución en el constructor** — `rank_invariant_check.py:141-145`: el string `"median"` (default) se resuelve a `np.median` y se guarda en `self._last_diff_strategy` (expuesto vía la property `last_diff_strategy`).
3. **Uso real** — `_maximum_abs_noises`, líneas **256-258**:
   ```python
   maximum_abs_noises.iloc[-1] = maximum_abs_noises.iloc[:-1].apply(
       self.last_diff_strategy
   )
   ```
   Es el único punto donde se ejecuta. La tabla `maximum_abs_noises` tiene una fila por alternativa no-óptima con la cota de ruido por criterio. La alternativa peor rankeada (última fila, sin "siguiente peor" con quién compararse) no tiene ese valor calculado por diferencia directa — se le asigna, columna por columna, la **mediana** de las cotas del resto de las alternativas. En `ws15`/`ws7` esa fila es **DOGE**.

## Fix aplicado

En `_mutate_dm` (`rank_invariant_check.py`), se acota `b` (la cota superior del ruido uniforme) al valor absoluto actual del criterio en la alternativa a mutar, para que el ruido pueda acercarse a cero pero nunca cruzarlo:

```python
# cap the noise bound so it can approach but never cross zero for
# the alternative being worsened (a criterion value should not
# flip sign just because it was made "worse")
max_abs_noise_without_sign_flip = df.loc[mutate].abs()
bounded_max_abs_noise = alternative_max_abs_noise.clip(
    upper=max_abs_noise_without_sign_flip
)

noise = 0  # all noises == 0
while np.all(noise == 0):  # at least we need one noise > 0
    noise = bounded_max_abs_noise.apply(
        lambda b: random.uniform(0, b)
    )
```

### Verificación

- `coso.py` y variantes con `windows_size=7` corren sin error, 5/5 veces cada una (antes fallaba 5/5).
- `pytest tests/ranksrev/test_rank_invariant_check.py -q` → 20/20 tests OK.
- Las 4 fallas de `pytest tests/ranksrev/` pertenecen a `test_rank_transitivity_check.py` (`AttributeError: module 'skcriteria.utils.dag_rank' has no attribute 'as_dag'`) y son preexistentes en `dev` — confirmado corriendo la suite con `git stash` (sin el fix) y viendo las mismas 4 fallas, no relacionadas con este cambio.

## Grafo de llamadas desde `evaluate()` (contexto)

```
evaluate(dm)
│
├─▶ dmaker.evaluate(dm)                       # ranking de referencia (pipeline externo)
│
├─▶ _add_mutation_info_to_rank(...)           # 1 vez, para el ranking de referencia
│
├─▶ _generate_mutations(dm, rrank, repeat, random)   ── generador
│      │
│      ├─▶ _maximum_abs_noises(dm, rank)      # 1 sola vez, antes del loop
│      │
│      └─▶ (por cada iteración × cada alternativa no-óptima)
│             _mutate_dm(dm, mutate, alternative_max_abs_noise, random)
│                yield (iteration, mutated, mdm, noise)
│
├─▶ (por cada tupla que produce el generador)
│      ├─▶ dmaker.evaluate(mdm)                # ranking de la mutación (pipeline externo)
│      │      └─▶ ... EntropyWeighter._weight_matrix -> entropy_weights()  # acá explotaba
│      └─▶ _add_mutation_info_to_rank(...)
│
├─▶ _opr(results)
├─▶ _rank_displacement(results)
│
└─▶ return RanksComparator(...)
```
