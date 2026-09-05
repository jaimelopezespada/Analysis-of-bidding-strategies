# Especificación del modelo — Optimizador de estrategias de oferta (v1)

> Documento de encargo para Claude Code. Define qué construir, con qué
> matemática, qué entradas/salidas y con qué stack. Pensado para un TFM de
> ingeniería industrial sobre estrategias de oferta en el mercado diario
> eléctrico (SDAC / EUPHEMIA).

---

## 1. Objetivo

Construir una herramienta en Python que, **dada la firma técnico-económica de
una tecnología de generación**, determine **qué tipo de orden de oferta le
conviene más** y con **qué parámetros**, comparando los tipos disponibles y
devolviendo un **ranking por beneficio** junto con sus parámetros óptimos y
gráficas comparativas.

La unidad se modela como **price-taker** (no influye en el precio marginal, que
es un dato exógeno). El producto de oferta y sus parámetros son la variable de
decisión.

## 2. Alcance

**Dentro de v1:**
- Tipos de orden: **órdenes simples**, **órdenes complejas escalables (SCO)** y
  **bloques (SBO)**.
- Lado: **venta** (generación).
- Resolución temporal **configurable: 24 (horaria) o 96 (cuartohoraria)** MTU.
- Modos de precio **determinista y estocástico**, seleccionables.
- Criterio de decisión **multicriterio configurable** (beneficio esperado,
  CVaR, probabilidad de casar) con pesos ajustables.
- Motor de optimización **Pyomo + HiGHS** (open-source; CBC como alternativa).

**Fuera de v1 (dejar interfaces preparadas para extender):**
- Bloques exclusivos (EXBO) y vinculados (LSBO) → v2.
- Lado de compra/demanda.
- Carteras multi-unidad (solo una tecnología por ejecución en v1).
- Precio-maker / modelos binivel.

## 3. Concepto matemático

### 3.1 Estructura de dos etapas

- **Primera etapa (here-and-now):** se eligen los parámetros de la oferta `θ`
  para un tipo de orden dado (precios de oferta, término fijo `TF`, perfil de
  volumen mínimo `MAV_t`, ratio mínimo `MAR`, etc.). **`θ` es único y común a
  todos los escenarios.**
- **Segunda etapa (por escenario `s`):** dado el precio realizado `λ_t^s`, se
  calcula la energía casada `q_t^s` según las reglas de aceptación del producto
  y el beneficio `Π^s`.

El modo determinista es el caso particular `|S| = 1`.

### 3.2 Notación

| Símbolo | Significado |
|---|---|
| `t ∈ T` | periodo de mercado (MTU); `T` = 24 o 96 |
| `s ∈ S` | escenario de precio, prob. `ρ_s`, `Σ ρ_s = 1` |
| `λ_t^s` | precio marginal de casación (dato), €/MWh |
| `C_t` | coste variable, €/MWh |
| `C_SU` | coste de arranque, € |
| `Q̄_t` | energía máxima disponible (previsión/capacidad), MWh |
| `Q_min` | mínimo técnico, MWh |
| `R` | límite de gradiente, MWh/periodo |
| `P^V_t` | precio variable declarado en la oferta, €/MWh |
| `TF` | término fijo (SCO), € |
| `MAV_t` | volumen mínimo de aceptación (SCO), MWh |
| `P^B`, `MAR` | precio del bloque (€/MWh) y ratio mínimo (SBO) |
| `q_t^s`, `x^s` | energía casada (MWh) y ratio de aceptación ∈[0,1] (variables) |

### 3.3 Reglas de aceptación por tipo de orden (segunda etapa)

**Orden simple** — cada periodo se casa de forma independiente:
```
q_t^s = Q̄_t   si  P^V_t ≤ λ_t^s   ;   0 en caso contrario
```

**SCO** — participa con `MAV_t` por periodo y condición de ingresos mínimos a
nivel de día (si no se cumple, la orden se retira entera en ese escenario):
```
q_t^s ≥ MAV_t · δ_t^s ,   q_t^s ≤ Q̄_t · δ_t^s ,   δ_t^s ∈ {0,1}
Σ_t λ_t^s · q_t^s  ≥  TF + Σ_t P^V_t · q_t^s        (condición de ingresos mínimos)
```

**SBO (bloque)** — indivisible y escalable hasta `MAR`; casa todo el perfil o
nada:
```
MAR · u^s ≤ x^s ≤ u^s ,   u^s ∈ {0,1} ,   q_t^s = x^s · Q̄_t
Σ_t (λ_t^s − P^B) · Q̄_t ≥ 0            (welfare del bloque ≥ 0 para casar)
```

### 3.4 Beneficio y criterio multicriterio

Beneficio por escenario:
```
Π^s = Σ_t (λ_t^s − C_t) · q_t^s − C_SU · u^s
```

Métricas:
- **Beneficio esperado:** `E[Π] = Σ_s ρ_s Π^s`
- **CVaR (Rockafellar–Uryasev), nivel α:** maximizar
  `CVaR_α = η − (1/(1−α)) · Σ_s ρ_s · z_s`,
  con `z_s ≥ η − Π^s`, `z_s ≥ 0`, `η` libre.
- **Probabilidad de casar:** `P_match = Σ_s ρ_s · 1[orden casada en s]`
  (usar las binarias de aceptación del producto).

Función objetivo configurable por pesos:
```
max  w1·E[Π] + w2·CVaR_α + w3·P_match
```
Pesos por defecto: `w1=1, w2=0, w3=0`. **El ranking final se ordena por
beneficio esperado** (columna principal), mostrando además el resto de métricas
y el valor del objetivo ponderado.

### 3.5 Linealización (mantener el problema como MILP)

Para evitar términos bilineales (precio de oferta × aceptación), **las
decisiones de oferta se eligen de una rejilla discreta de candidatos**
configurable (niveles de precio, valores de `MAR`, fracciones de `MAV`,
valores de `TF`). El modelo selecciona, mediante binarias, la combinación que
maximiza el objetivo. La aceptación por escenario se modela con las binarias de
3.3. Esto deja un **MILP** resoluble con HiGHS y, además, es más interpretable
para el TFM.

Para la SCO (el único producto cuya casación requiere solver) existen **dos
modelos de casación** seleccionables (`sco_model` en el run YAML):

- **`aware`** (por defecto, consistente con el mercado): para cada theta fijo,
  la casación de `(q, u, w)` maximiza el **excedente declarado**
  `Σ_s ρ_s Σ_t (λ_t^s − P^V)·q_t^s` — exactamente la información que EUPHEMIA
  ve (P^V, TF, MAV vía la condición de ingresos mínimos). Los costes privados
  del generador (`C`, `C_SU`) y el CVaR entran solo **después**, al evaluar
  `Π^s` sobre el despacho ya fijado y al seleccionar el mejor theta de la
  rejilla (igual que ya hacen Simple/SBO). Consecuencia: un periodo OTM
  (`P^V > λ_t^s`) nunca se despacha (contribución declarada negativa), y la
  única protección frente a días con pérdida real es la MIC declarada (TF,
  P^V) — por eso TF deja de ser trivialmente 0. Como el objetivo declarado es
  separable por hora, este modelo se resuelve en **forma cerrada** (despacho
  pleno en horas `λ ≥ P^V`; `u=1` si el excedente máximo cubre `TF`), sin
  solver (~4 órdenes de magnitud más rápido); su formulación MILP equivalente
  se conserva únicamente como cross-check en tests. Convención en el empate
  `λ = P^V` (contribución declarada nula): la hora se despacha, igual que en
  la orden simple.
- **`naive`** (benchmark heredado, cota de conocimiento perfecto): el MILP
  maximiza directamente `(1−β)·E[Π] + β·CVaR_α` con los costes reales dentro
  del objetivo, es decir, la casación conoce `C`/`C_SU` — algo que el mercado
  real no puede replicar. Se conserva para cuantificar el sesgo optimista del
  supuesto.

> Implementar también un cross-check por **enumeración** de la rejilla
> (evaluar cada combinación directamente sin solver) para validar que el MILP
> da el mismo óptimo en casos pequeños. El oráculo de enumeración corresponde
> a la forma cerrada del modelo `aware` sin MAV/mínimo técnico.
>
> Validación **out-of-sample** (`python -m bidding validate-oos`): theta* se
> optimiza sobre la fracción cronológica inicial de días (train) y se aplica
> mecánicamente, con la misma regla de aceptación declarada y sin
> re-optimizar, sobre los días restantes (test), para comprobar que el
> E[Π]/CVaR reportado es representativo de un despliegue real.

### 3.6 Restricciones técnicas opcionales

Activables según tecnología: mínimo técnico `q_t^s ≥ Q_min·u^s`, gradiente
`|q_t^s − q_{t-1}^s| ≤ R`, coste de arranque `C_SU` en el objetivo. En
renovables no programables (Familia 1) se desactivan.

## 4. Arquitectura del software

Módulos desacoplados, cada tipo de orden como una **estrategia** que comparte
una interfaz común (patrón strategy), de modo que añadir EXBO/LSBO en v2 sea
solo crear nuevas clases.

```
src/bidding/
  config.py        # modelos pydantic; carga y validación de YAML/JSON
  prices.py        # carga CSV de OMIE; construcción de escenarios
  technology.py    # dataclass/clase Technology
  orders/
    base.py        # clase abstracta OrderStrategy (build_model, accept_rule, params)
    simple.py
    sco.py
    sbo.py
  optimizer.py     # construye y resuelve el MILP (Pyomo) por tipo de orden
  metrics.py       # E[Π], CVaR, P_match, energía casada
  ranking.py       # ensambla la tabla comparativa y la ordena
  plots.py         # gráficas comparativas (matplotlib)
  cli.py           # interfaz de línea de comandos
```

## 5. Entradas

### 5.1 Configuración de tecnología (YAML que edita el usuario)

```yaml
name: "Solar FV 100 MW"
family: 1                 # 1..4
side: sell
variable_cost: 0.0        # €/MWh; escalar o lista de longitud T
# variable_cost_source: monthly_price_mean  # coste dinámico (hidráulica):
#   el coste se resuelve por mes como el precio medio mensual del mercado
#   (proxy del valor del agua); en ese caso se omite variable_cost.
price_reference: 10.0     # €/MWh — referencia de price_levels_pct; por defecto
                          # es variable_cost, OBLIGATORIA si el coste es 0 o lista.
startup_cost: 0.0         # €
technical_min: 0.0        # MWh (0 si no aplica)
ramp_limit: null          # MWh/periodo (null = sin límite)
availability:             # Q̄_t (previsión/capacidad por periodo)
  values: [0, 0, 0, 0, 0, 0, 0, 3, 5, 9, 10, 12, 12, 14, 12, 12, 11, 9, 4, 2, 1, 0, 0, 0]
```

En los bloques EXBO/LSBO, `availability` admite además
`source: tech_fraction`: los `values` pasan a ser fracciones horarias en
[0, 1] de la disponibilidad real (por escenario) de la tecnología, de modo
que la cantidad declarada del bloque sigue el recurso de cada día.

### 5.2 Precios (CSV histórico real de OMIE, aportado por el usuario)

Formato esperado (la herramienta debe validar y dar errores claros si no
encaja):

```
date,period,price
2025-03-18,1,45.2
2025-03-18,2,42.1
...
```
- `period` ∈ 1..T (24 o 96 según resolución).
- Cada **día** del CSV se trata como **un escenario** con probabilidad uniforme
  (configurable). En modo determinista se usa un único día o el día medio.
- Incluir un pequeño CSV de ejemplo en `data/` para poder ejecutar sin datos
  reales.

### 5.3 Configuración de ejecución (YAML)

```yaml
mode: stochastic          # deterministic | stochastic
resolution: 24            # 24 | 96
order_types: [simple, sco, sbo]
prices:
  csv_path: data/omie_2025.csv
  scenario_mode: per_day  # cada día = escenario
  date_range: ["2025-01-01", "2025-03-31"]   # opcional
objective:
  weights: {expected_profit: 1.0, cvar: 0.0, match_prob: 0.0}
  cvar_alpha: 0.95
candidate_grid:           # rejilla discreta de decisiones de oferta
  # price_levels_pct: FRACCIONES de la referencia de precio de la tecnología
  # (variable_cost o price_reference); absoluto = pct × referencia.
  price_levels_pct: [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
  mar_levels: [0.0, 0.3, 0.5, 0.8, 1.0]    # SBO
  mav_fraction_levels: [0.0, 0.5, 0.8, 1.0] # SCO: MAV como fracción de Q̄_t
  tf_levels: [0, 50, 100]                   # SCO: término fijo € (absoluto)
solver: highs             # highs | cbc
output_dir: results/
seed: 42
```

## 6. Pipeline de ejecución

1. Cargar y validar config de tecnología y de ejecución (pydantic).
2. Cargar CSV de OMIE → matriz de precios `λ_t^s` (escenarios × periodos).
2b. **Particionar los escenarios por mes calendario** (una optimización por
   mes): resolver el coste dinámico si procede (valor del agua = precio medio
   mensual) y la rejilla `price_levels_pct` → absoluta contra la referencia
   del mes; renormalizar probabilidades dentro del mes.
3. Para **cada mes** y **cada tipo de orden** en `order_types`:
   - construir el MILP (Pyomo) con la rejilla de candidatos y las reglas de
     aceptación de 3.3, objetivo de 3.4;
   - resolver con HiGHS;
   - extraer parámetros óptimos `θ*` y, por escenario, `q_t^s`, `Π^s`.
4. Calcular métricas (E[Π], CVaR, P_match, energía casada esperada).
5. Ensamblar el **ranking**, ordenado por beneficio esperado.
6. Generar tabla de salida y gráficas.

## 7. Salidas

### 7.1 Tabla de ranking (`results/ranking.csv` + impresión en consola)

Columnas:
`order_type, expected_profit, cvar, match_probability, expected_matched_energy,
objective_value, optimal_params (JSON)`

Ordenada por `expected_profit` descendente. `optimal_params` recoge los `θ*` de
cada producto (p. ej. SCO: `{P_V, TF, MAV_profile}`; SBO: `{P_B, MAR}`).

### 7.2 Gráficas comparativas (`results/figs/`)

- Barras: beneficio esperado por tipo de orden (con barra de CVaR superpuesta).
- Distribución del beneficio entre escenarios por tipo de orden (boxplot o
  histograma).
- Perfil de oferta óptima vs. perfil de precio medio (línea, por tipo de orden).
- Energía casada esperada por periodo (heatmap o área), por tipo de orden.

## 8. Stack tecnológico

- Python ≥ 3.11.
- **Pyomo** (modelado) + **HiGHS** (`highspy`) como solver MILP; CBC opcional.
- **pandas** (datos), **numpy**, **matplotlib** (gráficas), **pydantic**
  (validación de config), **PyYAML**.
- Gestión con `pyproject.toml` (o `requirements.txt`); entorno reproducible.
- Código tipado (type hints), docstrings, `ruff` + `black` para estilo.

## 9. Interfaz (CLI)

```
python -m bidding run --tech config/solar.yaml --run config/run.yaml
python -m bidding run --tech config/ccgt.yaml --run config/run.yaml --mode deterministic
```
Salida: escribe `results/ranking.csv`, las figuras y un resumen por consola con
el ranking.

## 10. Validación y tests

- Tests unitarios (pytest) de las reglas de aceptación de cada producto con
  casos a mano (precio constante, precio en escalón).
- Test de consistencia MILP vs. enumeración de la rejilla en instancias
  pequeñas (deben coincidir).
- Caso de regresión con el CSV de ejemplo: el ranking debe ser estable.
- Validación de entradas: longitudes de perfiles = `T`, periodos completos por
  día en el CSV, pesos del objetivo ≥ 0.

## 11. Supuestos y convenciones

- Price-taker: `λ_t^s` exógeno; la oferta no desplaza el precio.
- Misma oferta `θ` en todos los escenarios (es una decisión real de mercado).
- Empates de casación (precio de oferta = precio marginal): regla configurable,
  por defecto se acepta.
- Unidades coherentes: energía en MWh por MTU, precios en €/MWh, `TF`/`C_SU` en €.
- Resultados deterministas dado `seed`.

## 12. Decisiones abiertas (confirmar antes de codificar si surge duda)

- ¿`scenario_mode` adicional por *bootstrap* o por clústeres de días, además de
  `per_day`? (v1: solo `per_day`, dejar enganche.)
- ¿Precio de oferta de la SCO único para el día o por periodo? (v1 sugerido:
  único `P^V`, con `MAV_t` por periodo.)
- ¿Incluir ya un esqueleto de `report.py` (informe md) aunque no se pida en v1?
  (Opcional; las gráficas y la tabla son obligatorias.)
