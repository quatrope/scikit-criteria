# TODO: revisar boundary de validación de NaN/Inf (después de cerrar el bug de `RankInvariantChecker`)

## Contexto

Mientras se investigaba por qué `coso.py` fallaba con
`ValueError: The data [...] doesn't look like a ranking` (ver `explain.md` para
la cadena causal completa), surgió un punto de diseño más de fondo que **no
depende de `RankInvariantChecker`** y que sigue sin resolverse:

> El sistema deja circular `NaN`/`Inf` en silencio desde el punto donde se
> generan hasta muy lejos de ahí, en vez de fallar en el origen con un
> mensaje claro.

Esto se notó puntualmente con `EntropyWeighter`, pero el problema es general.

## Lo que ya se investigó (hechos, no propuestas)

Se revisaron los puntos de la cadena por donde pasa un `NaN`/`Inf` sin ser
detectado:

1. **`entropy_weights`** (`skcriteria/preprocessing/weighters.py:214-235`)
   asume que la matriz es no-negativa (trata cada columna como una
   distribución de probabilidad sin normalizar, se la pasa a
   `scipy.stats.entropy`). No valida esa precondición. Si hay un valor
   negativo en una columna, `scipy.stats.entropy` devuelve `-inf` para esa
   columna sin lanzar ningún warning propio de dominio, y la división final
   (`entropy_divergence / np.sum(entropy_divergence)`) produce pesos
   `inf`/`nan`.

2. **`SKCTransformerABC._transform_dm`**
   (`skcriteria/preprocessing/_preprocessing_base.py:51-67`) no valida nada:
   toma lo que devuelva `_transform_data` (en este caso los pesos rotos de
   `entropy_weights`) y arma un `DecisionMatrix` nuevo directo.

3. **`DecisionMatrix.__init__`** (`skcriteria/core/data.py:210-228`) solo
   valida que la cantidad de `weights`/`objectives`/columnas de `matrix`
   coincidan entre sí. **No hay ningún chequeo de `NaN`/`Inf`** ni en
   `matrix` ni en `weights`, en ningún punto de construcción o de
   `.replace(...)`.

4. Río abajo, **TOPSIS** termina devolviendo scores `nan`, **`rankdata`**
   (`skcriteria/utils/rank.py:65`) los castea a un entero centinela
   (`-9223372036854775808`) en vez de fallar ahí, y recién en
   **`_agg_base.py:428`** (`RankResult._validate_result`) se detecta que "no
   parece un ranking" — con un mensaje que no menciona nada de NaN, entropía,
   ni de dónde vino el problema.

## El punto de diseño a resolver

Aunque el bug puntual de `RankInvariantChecker` (el ruido de mutación podía
cruzar cero) se corrija, **cualquier otro camino** que le pase a
`EntropyWeighter` una matriz con un valor negativo — un dataset real con esa
forma, otro preprocesador que introduzca negativos antes en el pipeline,
etc. — va a producir el mismo `NaN` silencioso y el mismo error críptico
varias capas más abajo. El fix en `RankInvariantChecker` tapa un síntoma
puntual, no la causa de que el sistema tolere `NaN` circulando.

Dos lugares posibles donde cortar esto (no son excluyentes entre sí):

### Opción A — Validar en `entropy_weights` / `EntropyWeighter`

Si la matriz tiene negativos, lanzar un `ValueError` ahí mismo (ej.
`"EntropyWeighter requires non-negative matrix values"`).

- Pros: acotado, barato, específico — tiene sentido matemático (Shannon
  entropy sobre probabilidades no está definida para negativos), bajo blast
  radius.
- Contras: no protege contra otras fuentes futuras de `NaN`/`Inf` (otros
  weighters, scalers, métodos de agregación) — hay que repetir la validación
  en cada lugar que tenga una precondición similar.

### Opción B — Validar en `DecisionMatrix`

Rechazar cualquier `NaN`/`Inf` en `matrix` o `weights` al construirse
(constructor y/o `.replace(...)`).

- Pros: garantía general y mucho más fuerte — corta esta clase entera de bug
  en cualquier preprocesador presente o futuro, no solo `EntropyWeighter`.
  Es el punto más temprano posible donde se podría fallar rápido con un
  mensaje útil.
- Contras: blast radius mucho mayor — toca la clase más central de la
  librería. Riesgo de romper algún caso de uso legítimo que hoy dependa
  (aunque sea sin querer) de poder tener un `NaN` transitorio en algún punto
  intermedio del pipeline. Requiere revisar toda la suite de tests y
  probablemente varios preprocesadores/agregadores existentes.

**Pendiente de decisión del mantainer**: ¿A, B, o ambas? ¿En qué orden
conviene abordarlas?

## Cosas ya resueltas en esta sesión (para no repetir)

Estos cambios ya están aplicados en el working tree (no confirmados en un
commit todavía) y no forman parte de este TODO — se documentan acá solo para
que quede el contexto completo en un solo lugar:

- `skcriteria/ranksrev/rank_invariant_check.py`:
  - `_maximum_abs_noises`/`_mutate_dm` se extrajeron a funciones públicas de
    módulo (`maximum_abs_noises`, `mutate_dm`), siguiendo el mismo patrón que
    ya usa `weighters.py` (`entropy_weights` público + `_weight_matrix`
    privado). Los métodos de instancia quedaron como wrappers privados
    delgados.
  - En `mutate_dm` se acotó (`clip`) el límite superior del ruido uniforme al
    valor absoluto del criterio en la alternativa a mutar, para que el ruido
    pueda acercarse a cero pero nunca cruzarlo (evita el bug puntual descrito
    en `explain.md`).
  - Se sacó un `ipdb.set_trace()` que había quedado metido a mano en
    `maximum_abs_noises`.
- `tests/ranksrev/test_rank_invariant_check.py`: se agregó
  `test_RankInvariantChecker_mutation_noise_does_not_cross_zero`,
  parametrizado sobre 50 seeds, usando las funciones públicas de módulo
  directamente (sin pasar por `RankInvariantChecker`). 60/60 tests de este
  archivo pasan.

**Sin confirmar todavía**: no se corrió `coso.py` end-to-end después del
último cambio (se interrumpió antes de terminar), ni se corrió la suite
completa de `pytest tests/` fuera de `tests/ranksrev/`.
