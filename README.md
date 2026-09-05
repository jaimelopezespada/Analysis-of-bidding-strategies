# bidding — Optimizador de estrategias de oferta en el mercado diario eléctrico

Modelo y código de soporte del TFM *"Analysis of bidding strategies for various
technologies through the use of block orders, linked orders, exclusive block
orders and scalable complex orders"*. Dada una tecnología (firma de
restricciones técnico-económicas) y un conjunto de escenarios de precio, el
optimizador compara los distintos tipos de orden del SDAC —oferta simple,
orden compleja escalable (SCO), bloque simple (SBO), grupo exclusivo de
bloques (EXBO) y bloques vinculados (LSBO)— y determina, para cada uno, los
parámetros de oferta que maximizan `(1-β)·E[Π] + β·CVaR_α[Π]`.

La formulación matemática completa está en `SPEC_modelo_ofertas.md` y en
`TFM_estructura.tex` (sección Metodología/Implementación); este README se
limita a la organización del código y a cómo ejecutarlo.

## Instalación

Requiere Python ≥3.11.

```bash
pip install -e ".[dev]"
```

Instala el paquete `bidding` en modo editable (ver `pyproject.toml`) más las
dependencias de desarrollo (`pytest`, `ruff`, `black`). Las dependencias de
producción son `pandas`, `numpy`, `matplotlib`, `pydantic`, `PyYAML`, `pyomo`
y `highspy` (solver HiGHS, usado solo para el MILP de la SCO).

## Estructura del código

```
src/bidding/            paquete principal
├── config.py             modelos pydantic: TechnologyConfig, RunConfig,
│                          CandidateGrid, AvailabilityConfig, bloques EXBO/LSBO
├── prices.py              carga y valida el CSV de precios OMIE -> matriz λ_t^s (S,T)
├── availability.py        disponibilidad Q̄_t^s: estática (planta programable) o
│                          renovable real (ESIOS, escalada a la potencia nominal)
├── orders/                una clase Strategy por producto (patrón Strategy)
│   ├── base.py              interfaz común OrderStrategy.evaluate()
│   ├── simple.py             oferta simple — regla cerrada, sin solver
│   ├── sco.py                orden compleja escalable — el único que usa MILP
│   ├── sbo.py                bloque simple — cerrado (welfare ≥ 0)
│   ├── exbo.py                grupo exclusivo de bloques — cerrado (argmax welfare)
│   └── lsbo.py                bloques vinculados — cerrado (regla padre-hijo)
├── optimizer.py           construcción/resolución del MILP (Pyomo + HiGHS)
│                          para SCO; único módulo que necesita un solver real
├── metrics.py              E[Π], CVaR_α, probabilidad de casar, energía esperada
├── ranking.py              ensambla la tabla comparativa de tipos de orden
├── frontier.py             barrido del parámetro de aversión al riesgo β
├── family.py                agregación de resultados por familia tecnológica
├── seasons.py               comparación de una tecnología entre dos RunConfig
│                          (p. ej. verano vs invierno)
├── plots.py                 gráficas (beneficio, despacho, frontera, comparativas)
├── cli.py                   interfaz de línea de comandos (`python -m bidding`)
└── __main__.py              entry point (`python -m bidding ...`)

data/                    origen y descarga de datos
├── precios_omie.py        descarga precio mercado diario desde ESIOS (indicador 600)
├── generacion_renovable.py descarga generación real eólica/solar desde ESIOS
│                          (indicadores 10037 / 10205)
├── generate_example.py     genera un CSV sintético de precios para tests/demos
├── example_omie.csv        CSV de juguete (usado en tests y YAML de demo)
└── precios_omie_*.csv,     CSV reales descargados: verano 2025 e invierno
    generacion_renovable_*.csv  2025-2026, a resolución horaria

yaml/                    configuración declarativa
├── solar_fv.yaml, eolica.yaml, nuclear.yaml,  YAML de TECNOLOGÍA (TechnologyConfig)
│   hidraulica_embalse.yaml, ccgt.yaml,
│   carbon.yaml, bateria.yaml
└── run.yaml, run_verano.yaml, run_invierno.yaml,  YAML de EJECUCIÓN (RunConfig)
    run_frontier_demo.yaml, run_exbo_lsbo_demo.yaml,
    run_verano_exbo_demo.yaml

tests/                   pytest — unitarios por tipo de orden, consistencia
                         MILP-vs-enumeración, config, CLI, datos reales (marcados
                         `slow`)

src/eda_data/eda.py      análisis exploratorio (EDA) de precios/generación real
                         usado para las figuras de la memoria (Sección Metodología)

results/                 salida generada (rankings, gráficas, comparativas) — no
                         versionado como código fuente, se recrea al ejecutar

SPEC_modelo_ofertas.md   especificación técnica de referencia del modelo
TFM_estructura.tex       memoria del TFM (LaTeX)
```

### Arquitectura en una frase

`config.py` define **qué** se está modelando (tecnología + ejecución) →
`prices.py`/`availability.py` cargan **con qué datos** (matrices `(S,T)`) →
cada `orders/*.py` decide, para un tipo de orden, **cuánto se casa** dado un
candidato de parámetros de oferta (usando `optimizer.py` solo si hace falta
un MILP) → `metrics.py`/`ranking.py` resumen y comparan → `plots.py` grafica
→ `cli.py` conecta todo con la línea de comandos.

## Formas de ejecutar

### 1. Evaluar una tecnología (todos sus tipos de orden)

```bash
python -m bidding run --tech yaml/ccgt.yaml --run yaml/run_verano.yaml
```

Evalúa `--tech` (YAML de tecnología) con la configuración de `--run` (YAML de
ejecución: precios, rejilla de candidatos, β, tipos de orden a comparar...).
El run se **particiona por mes calendario**: una optimización completa por mes
(necesario porque el valor del agua de la hidráulica se resuelve por mes; ver
Notas). Imprime el ranking agregado por consola y guarda:

- `results/<season>/<tech>/<YYYY-MM>/ranking.csv` + `figs/` — resultados de
  cada mes;
- `results/<season>/<tech>/ranking_by_month.csv` — concatenado mensual con
  columnas `month`, `n_days` y `variable_cost` (el coste resuelto del mes);
- `results/<season>/<tech>/ranking.csv` — agregado de todo el periodo
  (beneficios de cada día bajo el θ* de su mes, ponderados por días).

El subdirectorio `<season>` solo aparece si el YAML de ejecución define
`season:`. Acepta `--mode deterministic|stochastic` para sobrescribir el modo
del YAML (en determinista se usa un día medio **por mes**).

Variantes útiles:

```bash
# Todas las tecnologías de yaml/ una tras otra, verano e invierno:
python -m bidding run --tech all \
    --run-verano yaml/run_verano.yaml --run-invierno yaml/run_invierno.yaml

# Cobrar un arranque por cada transición cero→producción (solo ofertas simples):
python -m bidding run --tech yaml/ccgt.yaml --run yaml/run_invierno.yaml \
    --startup-per-transition
```

- `--tech all` descubre todos los YAML de tecnología de `--yaml-dir` (por
  defecto `yaml/`) y ejecuta el pipeline completo (ranking + figuras) para
  cada uno de forma independiente; si una tecnología falla, se omite y el
  batch continúa. `--run-verano`/`--run-invierno` (alternativa a `--run`)
  ejecutan las dos temporadas de una vez — los resultados no chocan porque
  cada run define su `season:`.
- `--startup-per-transition` / `--no-startup-per-transition` sobrescriben el
  campo `startup_per_transition` del YAML de ejecución (por defecto `false`):
  en vez de restar un único arranque diario cuando hay despacho, la oferta
  simple resta `startup_cost` por cada vez que pasa de cero a producir (el
  arranque inicial incluido). Los demás tipos de orden no cambian, porque
  despachan bloques contiguos con a lo sumo un arranque.

### 2. Evaluar toda una familia tecnológica

```bash
python -m bidding family --family-num 3 --run yaml/run_verano.yaml
# o las 4 familias de una vez:
python -m bidding family --family-num all --run yaml/run_verano.yaml
```

Descubre automáticamente los YAML de tecnología cuyo campo `family:` coincide
(barriendo `yaml/`), evalúa cada uno y produce una comparativa conjunta en
`results/family_<n>/comparison.csv` + gráfica.

Variante estacional (evalúa cada tecnología de la familia en verano **e**
invierno y etiqueta cada fila con la temporada):

```bash
python -m bidding family --family-num 3 \
    --run-verano yaml/run_verano.yaml --run-invierno yaml/run_invierno.yaml
```

### 3. Comparar una tecnología entre dos configuraciones (p. ej. estacional)

```bash
python -m bidding compare-seasons --tech yaml/solar_fv.yaml \
    --run-verano yaml/run_verano.yaml --run-invierno yaml/run_invierno.yaml
```

Evalúa la misma tecnología contra dos `RunConfig` cualesquiera (no tiene por
qué ser verano/invierno) y guarda
`results/comparacion_estacional/<tech>/comparison.csv` + gráfica.

### 4. Demos puntuales (EXBO/LSBO, frontera de riesgo)

```bash
python -m bidding run --tech yaml/solar_fv.yaml --run yaml/run_exbo_lsbo_demo.yaml
python -m bidding run --tech yaml/ccgt.yaml     --run yaml/run_frontier_demo.yaml
```

`run_exbo_lsbo_demo.yaml`/`run_verano_exbo_demo.yaml` solo funcionan con
tecnologías que declaren `exbo_groups`/`lsbo_families` en su propio YAML (por
ahora `solar_fv.yaml` y `bateria.yaml`). `run_frontier_demo.yaml` activa
`beta_sweep`, que genera además `frontier.csv` y la gráfica de la frontera
beneficio-riesgo (E[Π] vs CVaR).

### 5. Tests

```bash
pytest                    # suite completa
pytest -m "not slow"      # excluye los tests que cargan los CSV reales de verano/invierno
```

### 6. Regenerar los datos de entrada (opcional)

Los CSV de `data/` ya están versionados; solo hace falta volver a descargarlos
si se quiere ampliar el rango de fechas. Requiere un token personal de la API
de ESIOS:

```python
from data.precios_omie import obtener_precios_mercado_diario
from data.generacion_renovable import ...  # ver docstrings de cada script
```

`data/generate_example.py` regenera el CSV sintético `example_omie.csv` usado
en tests y en los YAML de demo (no requiere API key).

### 7. EDA de la memoria

```bash
python -m src.eda_data.eda
```

Compara verano 2025 vs invierno 2025-2026 (perfil intradiario, distribución de
precios, volatilidad diaria, correlación precio-renovable) y escribe las
figuras en `results/eda/figs/`, las mismas que ilustran la Sección
Metodología del TFM.

## Notas

- **Namespacing por temporada**: `RunConfig.season` (definido en
  `run_verano.yaml`/`run_invierno.yaml`) evita que dos ejecuciones con el
  mismo `output_dir` se pisen entre sí — ver `cli.tech_output_dir`.
- **Rejilla por tecnología**: si un YAML de tecnología define su propio
  `candidate_grid`, sobrescribe la rejilla genérica del YAML de ejecución
  (`TechnologyConfig.resolved_grid`).
- **Niveles de precio relativos (`price_levels_pct`)**: los candidatos de
  precio de todos los YAML son FRACCIONES de una referencia por tecnología
  (`absoluto = pct × referencia`; 1.0 = ofertar exactamente a la referencia).
  La referencia es el `variable_cost` de la tecnología; con coste 0 (solar,
  eólica) es obligatorio declarar `price_reference` explícito. Así, si cambia
  el coste de una tecnología no hay que reescalar ninguna rejilla a mano.
- **Valor del agua mensual (hidráulica)**: `variable_cost_source:
  monthly_price_mean` hace que el coste variable — proxy del coste de
  oportunidad del agua — se resuelva en ejecución como el precio medio
  mensual del mercado (media de todos los precios horarios de los escenarios
  del mes; ver `monthly.py`). Por eso el run se particiona por mes: cada mes
  optimiza con su valor del agua y su rejilla pct reescalada. Limitación
  documentada: calcular la media del propio mes implica previsión perfecta
  intra-mes.
- **Bloques como fracción del recurso (`source: tech_fraction`)**: en los
  bloques EXBO/LSBO de solar/eólica, `availability.values` son fracciones
  horarias en [0, 1] de la disponibilidad renovable real de cada día, de modo
  que la energía declarada del bloque sigue la estacionalidad del recurso en
  vez de prometer un perfil estático imposible en días de poco recurso.
- **Solver**: solo SCO resuelve un MILP (Pyomo + HiGHS); el resto de
  productos tienen solución cerrada — ver el docstring de `optimizer.py` para
  el razonamiento completo.
