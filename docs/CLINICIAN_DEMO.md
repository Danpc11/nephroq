# Guía de demostración para clínicos — NephroQ
### Clinician demo guide (ES + EN reference)

Guion completo de ~7 minutos para mostrar la utilidad del gemelo digital a un
nefrólogo, endocrinólogo o médico de primer contacto, en el contexto de
diabetes tipo 2 → enfermedad renal crónica (ERC).

Este documento es **autocontenido**: incluye los cuatro pacientes de ejemplo con
sus valores exactos y las salidas que debes esperar, las dos formas de correr la
demo, el guion hablado, las advertencias honestas y el cierre. No necesitas
abrir ningún otro archivo para dar la demo.

> ⚠️ **NephroQ es un prototipo de investigación. NO es una herramienta
> diagnóstica.** No debe usarse para decisiones clínicas sin supervisión médica
> calificada. Las trayectorias son *ilustrativas del mecanismo del modelo*, no
> predicciones individualizadas. Esto está escrito en la propia interfaz y no se
> puede desactivar.

---

## Índice

1. [Preparación](#1-preparación-2-minutos-una-sola-vez)
2. [La idea en una frase](#2-la-idea-en-una-frase-para-arrancar)
3. [Los cuatro pacientes de ejemplo](#3-los-cuatro-pacientes-de-ejemplo-tabla-de-referencia)
4. [El recorrido guiado](#4-el-recorrido-guiado-5-minutos)
5. [Las tres advertencias honestas](#5-las-tres-advertencias-honestas-dilas-en-voz-alta)
6. [El cierre: ¿para qué sirve?](#6-el-cierre-y-esto-para-qué-me-sirve)
7. [Si te piden ver los números](#7-si-te-piden-ver-los-números-del-modelo)
8. [Preguntas frecuentes](#8-preguntas-que-te-van-a-hacer)

---

## 1. Preparación (2 minutos, una sola vez)

Hay **dos formas** de dar la demo. Ambas comparten el mismo motor y los mismos
pacientes de ejemplo (definidos una sola vez en `src/i18n.py`), y ambas son
**bilingües** (inglés / español) con un selector de idioma.

### Opción A — App web (recomendada para consultorio o proyector)

```bash
git clone https://github.com/Danpc11/nephroq.git
cd nephroq
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app_web.py
```
Se abre en el navegador (`http://localhost:8501`). El selector de idioma está
arriba en la barra lateral.

### Opción B — Notebook de Colab (sin instalar nada)

Abre `risk_notebook.ipynb` en Google Colab → **Entorno de ejecución → Ejecutar
todo**. El código está oculto: el clínico solo ve los controles. Útil para
compartir por link, incluso desde un celular.

### Opción C — Desplegar en la nube

Sube el repo a Streamlit Community Cloud o Hugging Face Spaces (gratis, ver
[`WEB_DEPLOYMENT.md`](WEB_DEPLOYMENT.md)) y comparte la URL. Ideal para que el
clínico lo explore por su cuenta después de la demo.

### Nota sobre la calibración

En modo demostración la app usa una **calibración pública** (sintética +
comprobación de validez aparente contra Al-Shamsi 2018). Es suficiente para
mostrar el *comportamiento del modelo*; la app lo marca con un aviso amarillo.
**No presentes los números como pronósticos reales.** Si cargaste una
calibración local con MIMIC-IV y el *quality gate* la marcó como `warning`, la
app mostrará una advertencia roja — eso es intencional, ver
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

---

## 2. La idea en una frase (para arrancar)

> "En diabetes, el riñón no se deteriora de forma lineal: cuando se pierden
> nefronas, las que quedan se sobrecargan (**hiperfiltración**) y se dañan más
> rápido. NephroQ modela ese circuito de retroalimentación con un solo parámetro
> físico, **q**, que dice *qué tan abrupto* es el colapso final. Por eso un
> paciente puede verse estable en una foto puntual y aun así ir camino a
> diálisis."

---

## 3. Los cuatro pacientes de ejemplo (tabla de referencia)

Estos perfiles se cargan **con un clic** desde la barra lateral (app) o desde los
botones (notebook). Son perfiles **ilustrativos, no pacientes reales**.

Salidas esperadas con la **calibración v2 anclada a ensayos** (progresión fijada por
los brazos placebo de CREDENCE y EMPA-KIDNEY; DAPA-CKD predicho fuera de muestra):

| Paciente | Edad/Sexo | Cr | HbA1c | UACR | PAS | **eGFR basal** | **KDIGO** | **Sin tx** | **Con reno** |
|---|---|---|---|---|---|---|---|---|---|
| 🔴 Progresor rápido | 58 F | 1.3 | 8.1 % | 145 | 142 | **47.7** | G3a | 13.4 años | >15 años |
| 🟠 eGFR normal, riesgo oculto | 49 M | 0.95 | 9.2 % | 280 | 150 | **98.1** | **G1** | >15 años | >15 años |
| 🟢 Bien controlado, bajo riesgo | 55 F | 1.0 | 6.6 % | 15 | 125 | **66.5** | G2 | >15 años | >15 años |
| 🟣 Avanzado (G3b–G4) | 63 M | 2.1 | 8.8 % | 600 | 155 | **34.7** | G3b | 7.2 años | >15 años |

### 🆕 La albuminuria predicha (el nuevo panel de la app)

En v2 la UACR es una **salida del modelo**, no una entrada fija. La app ahora
grafica su trayectoria — y para varios pacientes **esta es la historia principal**,
más que el tiempo al umbral:

| Paciente | UACR sin tratamiento (0 → 15 a) | UACR con renoprotección |
|---|---|---|
| 🔴 Progresor rápido | 145 → **760** | 104 → 183 |
| 🟠 **eGFR normal, riesgo oculto** | 280 → **1636** | 200 → 341 |
| 🟢 Bien controlado | 15 → 27 | 11 → 14 |
| 🟣 Avanzado | 600 → **6693** | 428 → 984 |

El modelo predice además una **caída inmediata de ~29 %** en la UACR al iniciar
tratamiento — los ensayos de iSGLT2 publicaron **31–35 %**. Ese número **no es un
ajuste**: sale de parámetros anclados en CREDENCE y verificados fuera de muestra
en DAPA-CKD.

> **⚠️ Cambio importante respecto a versiones previas de esta guía.** Con el modelo
> v1 el "progresor rápido" cruzaba el umbral en 5.0 años y el de "riesgo oculto" en
> 10.4. Esos números eran **demasiado agresivos**: la validación in-silico mostró que
> v1 hacía declinar a los pacientes **~2× más rápido** que los brazos placebo reales
> de los ensayos. Las cifras de v2 son las clínicamente plausibles (≈2–3 mL/min/año
> en un G3a). **Si tenías memorizado el guion viejo, actualízalo.**

> **Ojo con el lenguaje:** eGFR<15 es un **umbral modelado de función renal**, no
> una predicción de cuándo iniciaría realmente la diálisis. El inicio real
> depende de síntomas, laboratorios y juicio clínico. Nunca digas "el modelo
> predice que se dializa en 5 años".

---

## 4. El recorrido guiado (5 minutos)

Este es el orden que mejor cuenta la historia. **No empieces por el progresor
obvio** — empieza por el caso que una foto puntual no ve.

### 🟠 Paso 1 — "eGFR normal, riesgo oculto" *(empieza aquí)*

- **Qué verá el clínico:** eGFR basal **98.1**, categoría **G1** — "riñón normal"
  en un laboratorio de rutina. Pero HbA1c 9.2 %, UACR 280 mg/g, PAS 150.
- **Pausa aquí.** Deja que el clínico note que el eGFR está normal. Pregúntale:
  *"¿este paciente te preocupa?"*
- **Luego baja al panel de albuminuria.** Ahí está la historia: sin tratamiento el
  modelo proyecta la UACR de **280 → 1636 mg/g**; con renoprotección, 200 → 341.
  La curva de eGFR desciende pero no cruza el umbral en 15 años — **y eso está
  bien**: el daño se manifiesta primero como albuminuria progresiva, años antes de
  que el eGFR se desplome.
- **El mensaje:** *una sola medición de eGFR lo habría tranquilizado.* El mecanismo
  ya está en marcha y es visible en la albuminuria. Esta es la utilidad central:
  modelar el **mecanismo**, no solo el número de hoy.

### 🔴 Paso 2 — "Progresor rápido"

- **Qué verá:** G3a ya establecido (eGFR 47.7); cruce del umbral en **5.0 años**.
- **El mensaje:** compara las dos curvas. La renoprotección mueve el umbral a
  **7.9 años** — una diferencia de **~3 años**, no de meses. El caso "actuar
  ahora".

### 🟢 Paso 3 — "Bien controlado, bajo riesgo"

- **Qué verá:** trayectoria lenta, casi plana (12.6 años sin tratamiento).
- **El mensaje:** el modelo **no exagera**. Tan importante como detectar
  progresores es no alarmar a quien va bien. Es una comprobación de
  **especificidad / validez aparente** — y es la que le da credibilidad al
  paciente 🟠.

### 🟣 Paso 4 — "Avanzado (G3b–G4)"

- **Qué verá:** régimen de **colapso terminal**, donde el exponente **q** domina
  (2.4 años sin tratamiento).
- **El mensaje:** cerca del final, pequeñas diferencias de control se traducen en
  meses. Ilustra por qué importa la *forma* de la curva y no una pendiente lineal.

### 🔬 Remate — qué vale la pena medir

El mensaje accionable no es un examen caro, es el **tiempo de seguimiento**. Las
mismas 4–6 creatininas repartidas en 4–8 años estiman la tasa de progresión del
paciente ~3× mejor que esas mismas mediciones apretadas en 1–2 años — y mejor que
10–14 mediciones dentro de una ventana corta. La cistatina C ayuda de forma
modesta a la tasa individual (R² 0.67 vs 0.48 con solo creatinina), pero **no**
identifica el exponente de colapso `q` a nivel de un paciente: eso requiere una
población, no un examen. Recomendación para llevar hoy: rescata las creatininas
viejas del expediente — son gratis y valen más que un análisis nuevo.

---

## 5. Las tres advertencias honestas (dilas en voz alta)

Un buen clínico va a preguntar esto. **Adelántate.** Decirlo suma credibilidad:
demuestra que el modelo conoce sus límites. (Detalle completo en
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).)

1. **No está validado en cohorte clínica prospectiva.** Está verificado en datos
   sintéticos y con una primera comprobación de validez aparente contra perfiles
   reales publicados. Lo que falta: calibración y validación externa
   longitudinal, réplica *in-silico* de un ensayo (DAPA-CKD / CREDENCE / FLOW), y
   comparación cabeza a cabeza contra la ecuación estándar de riesgo (**KFRE**).

2. **La proyección asume marcadores constantes.** La app parte de la foto que
   tecleas y la mantiene fija hacia el futuro. El motor sí soporta covariables
   variables en el tiempo, pero la interfaz aún proyecta desde un solo punto. Si
   el control del paciente cambia, la trayectoria real cambia.

3. **La banda de incertidumbre es de *parámetros*, no un intervalo de
   predicción.** Captura la incertidumbre de calibración (bootstrap de
   pacientes), **no** el ruido de medición, ni la variabilidad individual, ni el
   futuro desconocido de los laboratorios. Por eso la app nunca la llama
   "intervalo de predicción".

---

## 6. El cierre: "¿y esto para qué me sirve?"

- **Estratificación temprana:** marca al paciente con eGFR normal pero
  hiperfiltrando, antes de que la caída sea visible en el laboratorio (el caso 🟠).
- **Conversación con el paciente:** una curva "con vs sin tratamiento" comunica el
  beneficio de la renoprotección mucho mejor que un valor aislado.
- **Qué medir:** justifica pedir **UACR** y **cistatina C** por su impacto
  concreto en la calidad de la predicción.
- **Investigación:** en el contexto IMSS–ISSSTE, es la base para una calibración
  con cohorte mexicana real y una comparación cabeza a cabeza contra KFRE.

**Lo que NO hay que afirmar:**
- ❌ que predice la fecha de inicio de diálisis;
- ❌ que sustituye el juicio clínico;
- ❌ que MIMIC-IV, por ser "datos reales", lo vuelve una herramienta validada.

---

## 7. Si te piden ver "los números" del modelo

Pipeline completo con evidencia auditable por corrida:
```bash
cd src
python insilico_trial.py    # replica in-silico contra 3 ensayos publicados
```

MVP de calibración/validación (ajuste, forecast vs extrapolación lineal,
discriminación de progresores):
```bash
cd src
python mvp_calibration.py                                  # datos sintéticos
CKD_CSV=../data/mis_datos.csv python mvp_calibration.py    # con tus datos
```
Formato del CSV: `patient_id, time_years, egfr, hba1c, uacr, sbp`.

Genera `results/validation_report.md` — el reporte de una página pensado para
compartir con colaboradores clínicos.

---

## 8. Preguntas que te van a hacer

**"¿En qué se diferencia de KFRE?"**
KFRE es una ecuación de riesgo estadística (probabilidad de falla renal a 2/5
años). NephroQ es **mecanístico**: simula la pérdida de nefronas y la
hiperfiltración, así que produce una *trayectoria* y permite simular
contrafactuales ("¿y si controlo la presión?"). La comparación cabeza a cabeza
contra KFRE está pendiente y es el siguiente hito — dilo así, sin adornos.

**"¿Qué es q exactamente?"**
El exponente de retroalimentación de la hiperfiltración: gobierna qué tan
abruptamente se acelera el daño cuando quedan pocas nefronas. Con q alto, la
curva se ve estable mucho tiempo y luego cae en picada.

**"¿Puedo meter a mis pacientes?"**
Sí, tecleando sus marcadores — pero con la calibración pública los números son
ilustrativos. Para que sea útil de verdad en población mexicana hace falta
recalibrar con una cohorte local (sección 7).

**"¿Por qué el eGFR<15 y no diálisis?"**
Porque es lo que el modelo puede sostener: un umbral de función renal. El inicio
de diálisis es una decisión clínica, no una línea en una curva.

---

*Regla de oro de la demo: muestra el paciente 🟠 primero, deja que el clínico
note que "el eGFR está normal", y solo entonces enseña la curva. Ese momento —
cuando la foto tranquiliza pero el mecanismo no— es toda la propuesta de valor.*
