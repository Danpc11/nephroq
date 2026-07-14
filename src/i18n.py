"""
================================================================================
i18n  ·  Single source of truth for UI strings (English / Spanish)
================================================================================
Both the Streamlit web app (app_web.py) and the Colab notebook import their
visible text and their example-patient vignettes from here, so the two
interfaces can never drift out of sync.

Usage:
    from i18n import t, PRESETS, LANGUAGES
    t("es", "title")                       -> Spanish title
    t("en", "diff_info", label="...", d=3) -> formatted English string
================================================================================
"""

LANGUAGES = {"English": "en", "Español": "es"}

# Placeholders in braces {like_this} are filled by t(...) via str.format.
STRINGS = {
    # ---- page / header -------------------------------------------------------
    "title": {
        "en": "🩺 NephroQ — renal risk digital twin in type 2 diabetes",
        "es": "🩺 NephroQ — gemelo digital de riesgo renal en diabetes tipo 2",
    },
    "disclaimer": {
        "en": "Research prototype (TRL4) — NOT a diagnostic tool. Must not be used "
              "for clinical decisions without qualified medical supervision.",
        "es": "Prototipo de investigación (TRL4) — NO es una herramienta diagnóstica. "
              "No debe usarse para decisiones clínicas sin supervisión médica calificada.",
    },
    "research_use": {
        "en": "**Research-use calibration — not externally validated.**",
        "es": "**Calibración de uso en investigación — no validada externamente.**",
    },
    "active_calibration": {
        "en": "Active calibration: **{src}**",
        "es": "Calibración activa: **{src}**",
    },
    "demo_mode": {
        "en": "**Demonstration mode** — projections use a research calibration anchored to "
              "PUBLISHED AGGREGATE TRIAL DATA (the placebo arms of CREDENCE and EMPA-KIDNEY; "
              "DAPA-CKD is held out and predicted). This is NOT patient-level clinical "
              "validation, and projections must not be read as individualized clinical "
              "predictions.",
        "es": "**Modo demostración** — las proyecciones usan una calibración de investigación "
              "anclada a DATOS AGREGADOS PUBLICADOS DE ENSAYOS (los brazos placebo de CREDENCE "
              "y EMPA-KIDNEY; DAPA-CKD se retiene y se predice). Esto NO es una validación "
              "clínica a nivel de paciente, y las proyecciones no deben leerse como "
              "predicciones clínicas individualizadas.",
    },
    "quality_warning": {
        "en": "**Calibration quality warning** — the active MIMIC-IV calibration was "
              "flagged by calibrate_mimic.py as unreliable: {reasons}. Do not treat "
              "these projections as trustworthy.",
        "es": "**Aviso de calidad de calibración** — calibrate_mimic.py marcó la "
              "calibración MIMIC-IV activa como poco confiable: {reasons}. No trate "
              "estas proyecciones como confiables. Ver the README (Limitations).",
    },
    # ---- sidebar -------------------------------------------------------------
    "quality_optin": {
        "en": "Use the flagged MIMIC calibration anyway (research mode, at your own risk)",
        "es": "Usar de todos modos la calibración MIMIC marcada (modo investigación, bajo tu responsabilidad)",
    },
    "fell_back_public": {
        "en": "**Falling back to the public calibration.** The MIMIC calibration on this machine "
              "was flagged as unreliable, so it is **not** being used for the projections below. "
              "Tick the box above to override (research only).",
        "es": "**Usando la calibración pública.** La calibración MIMIC de esta máquina fue marcada "
              "como poco confiable, así que **no** se está usando para las proyecciones de abajo. "
              "Marca la casilla de arriba para anular (solo investigación).",
    },
    "using_flagged": {
        "en": "**Research mode: using a calibration that FAILED its quality gate.** The projections "
              "below are not trustworthy and must not be shown to clinicians as results.",
        "es": "**Modo investigación: se está usando una calibración que NO pasó el control de calidad.** "
              "Las proyecciones de abajo no son confiables y no deben mostrarse a clínicos como resultados.",
    },
    "language": {"en": "Idioma / Language", "es": "Idioma / Language"},
    "examples_header": {"en": "Example patients", "es": "Pacientes de ejemplo"},
    "examples_caption": {
        "en": "Click to load an illustrative profile (not a real patient).",
        "es": "Haz clic para cargar un perfil ilustrativo (no es un paciente real).",
    },
    "reset": {"en": "↺ Reset to defaults", "es": "↺ Restablecer valores"},
    "markers_header": {"en": "Patient markers", "es": "Marcadores del paciente"},
    "age": {"en": "Age (years)", "es": "Edad (años)"},
    "sex": {"en": "Sex", "es": "Sexo"},
    "blood": {"en": "Blood", "es": "Sangre"},
    "creatinine": {"en": "Serum creatinine (mg/dL)", "es": "Creatinina sérica (mg/dL)"},
    "have_cystatin": {
        "en": "I have cystatin C (more precise)",
        "es": "Tengo cistatina C (más preciso)",
    },
    "cystatin": {"en": "Cystatin C (mg/L)", "es": "Cistatina C (mg/L)"},
    "hba1c": {"en": "HbA1c (%)", "es": "HbA1c (%)"},
    "urine": {"en": "Urine", "es": "Orina"},
    "uacr": {
        "en": "UACR — urine albumin/creatinine ratio (mg/g)",
        "es": "UACR — cociente albúmina/creatinina urinaria (mg/g)",
    },
    "in_clinic": {"en": "In clinic", "es": "En consulta"},
    "sbp": {"en": "Systolic blood pressure (mmHg)", "es": "Presión arterial sistólica (mmHg)"},
    "treated": {
        "en": "Already receiving a renoprotective therapy "
              "(illustrative SGLT2i-like intervention scenario)",
        "es": "Ya recibe terapia renoprotectora "
              "(escenario ilustrativo de una intervención tipo iSGLT2)",
    },
    # ---- main panel ----------------------------------------------------------
    "method_cr_cys": {
        "en": "creatinine + cystatin (more precise)",
        "es": "creatinina + cistatina (más preciso)",
    },
    "method_cr": {"en": "creatinine only", "es": "solo creatinina"},
    "example_loaded": {
        "en": "**Example loaded — {label}.** {note}",
        "es": "**Ejemplo cargado — {label}.** {note}",
    },
    "baseline_egfr": {"en": "Baseline eGFR", "es": "eGFR basal"},
    "baseline_help": {
        "en": "Calculated with: {method}",
        "es": "Calculado con: {method}",
    },
    "kdigo": {"en": "KDIGO GFR category", "es": "Categoría KDIGO de FG"},
    "label_current_tx": {"en": "Current treatment", "es": "Tratamiento actual"},
    "label_no_tx": {
        "en": "No treatment (current scenario)",
        "es": "Sin tratamiento (escenario actual)",
    },
    "label_reno_added": {
        "en": "Illustrative renoprotective scenario added",
        "es": "Escenario renoprotector ilustrativo añadido",
    },
    "label_tx_stopped": {"en": "If treatment is stopped", "es": "Si se suspende el tratamiento"},
    "time_title": {
        "en": "Modeled time to eGFR<15 ({state})",
        "es": "Tiempo modelado a eGFR<15 ({state})",
    },
    "state_current": {"en": "current", "es": "actual"},
    "state_untreated": {"en": "untreated", "es": "sin tratar"},
    "years": {"en": "{v:.1f} years", "es": "{v:.1f} años"},
    "gt_years": {"en": ">{v} years", "es": ">{v} años"},
    "time_help": {
        "en": "This is a modeled kidney-function threshold (eGFR<15), not a prediction "
              "of when dialysis would actually start. Real dialysis initiation depends "
              "on symptoms, labs, and clinical judgment.",
        "es": "Es un umbral modelado de función renal (eGFR<15), no una predicción de "
              "cuándo iniciaría realmente la diálisis. El inicio real depende de "
              "síntomas, laboratorios y juicio clínico.",
    },
    "boot_reach": {
        "en": "Of {n} bootstrap parameter resamples, **{pct:.0f}%** reach eGFR<15 "
              "within the {horizon}-year horizon shown.",
        "es": "De {n} remuestreos bootstrap de parámetros, **{pct:.0f}%** alcanzan "
              "eGFR<15 dentro del horizonte de {horizon} años mostrado.",
    },
    "boot_interval": {
        "en": "90% bootstrap **parameter**-uncertainty interval, among resamples that "
              "cross the threshold: **{lo:.1f} – {hi:.1f} years**. This reflects "
              "calibration-parameter uncertainty only — not measurement noise, "
              "individual variability, or unknown future lab values.",
        "es": "Intervalo de incertidumbre de **parámetros** bootstrap al 90%, entre los "
              "remuestreos que cruzan el umbral: **{lo:.1f} – {hi:.1f} años**. Refleja "
              "solo la incertidumbre de los parámetros de calibración — no el ruido de "
              "medición, la variabilidad individual ni valores futuros desconocidos "
              "(ver the README (Limitations)).",
    },
    "boot_no_interval": {
        "en": "Fewer than half of the bootstrap resamples reach the threshold within this "
              "horizon, so no interval is shown here. Modeled time is best read as "
              "'>{horizon} years' for a majority of parameter resamples.",
        "es": "Menos de la mitad de los remuestreos bootstrap alcanzan el umbral en este "
              "horizonte, por lo que no se muestra intervalo. El tiempo modelado se lee "
              "mejor como '>{horizon} años' para la mayoría de los remuestreos.",
    },
    "boot_degenerate": {
        "en": "**Uncertainty band hidden — this calibration is not trustworthy.** Every "
              "bootstrap resample returned the *same* parameters. That is not precision: it "
              "means the fitting procedure never actually moved, so its numbers carry no "
              "information. Drawing a narrow band here would look confident while being "
              "meaningless. A point estimate is shown instead. Re-run the calibration before "
              "relying on it.",
        "es": "**Banda de incertidumbre oculta — esta calibración no es confiable.** Todos los "
              "remuestreos bootstrap devolvieron los *mismos* parámetros. Eso no es precisión: "
              "significa que el procedimiento de ajuste nunca se movió, así que sus números no "
              "aportan información. Dibujar una banda estrecha aquí parecería seguro pero no "
              "significaría nada. Se muestra sólo la estimación puntual. Vuelve a correr la "
              "calibración antes de confiar en ella.",
    },

    "boot_none": {
        "en": "No bootstrap parameter-uncertainty band available for this calibration — "
              "point estimate only.",
        "es": "No hay banda de incertidumbre bootstrap para esta calibración — solo "
              "estimación puntual (ver the README (Limitations)).",
    },
    # ---- plot ----------------------------------------------------------------
    "band_label": {
        "en": "90% bootstrap parameter-uncertainty band",
        "es": "Banda de incertidumbre de parámetros bootstrap 90%",
    },
    "plot_threshold": {"en": "modeled eGFR<15 threshold", "es": "umbral modelado eGFR<15"},
    "plot_x": {"en": "years", "es": "años"},
    "plot_y": {
        "en": "projected eGFR (mL/min/1.73m²)",
        "es": "eGFR proyectado (mL/min/1.73m²)",
    },
    "plot_title": {
        "en": "Illustrative model projection of renal function",
        "es": "Proyección ilustrativa del modelo de la función renal",
    },
    "diff_info": {
        "en": "**{label}** changes the modeled time to the eGFR<15 threshold by "
              "approximately **{d:.1f} years** relative to the current scenario, under "
              "the assumptions of the current research model. This is not a prediction "
              "of dialysis initiation.",
        "es": "**{label}** cambia el tiempo modelado al umbral eGFR<15 en "
              "aproximadamente **{d:.1f} años** respecto al escenario actual, bajo los "
              "supuestos del modelo de investigación actual. Esto no es una predicción "
              "del inicio de diálisis.",
    },
    "cystatin_warning": {
        "en": "eGFR was calculated with creatinine only. If cystatin C is unavailable (as it "
              "often is), you do **not** need it: in simulation experiments, **repeating the "
              "creatinine at each visit and pulling older results from the chart beat cystatin "
              "C alone**. What matters most is the TIME SPAN of the history, not the number of "
              "measurements — the same 5 values spread over 6 years are worth far more than 12 "
              "crammed into 18 months. Note this improves the estimate of the patient's injury "
              "RATE; the collapse exponent q stays weakly identifiable either way.",
        "es": "El eGFR se calculó solo con creatinina. Si no hay cistatina C (lo habitual), **no "
              "la necesitas**: en experimentos de simulación, **repetir la creatinina en cada "
              "visita y rescatar resultados viejos del expediente supera a la cistatina C "
              "sola**. Lo que más importa es el LAPSO del historial, no el número de mediciones "
              "— 5 valores repartidos en 6 años valen mucho más que 12 apretados en 18 meses. "
              "Esto mejora la estimación de la TASA de daño del paciente; el exponente de "
              "colapso q sigue siendo débilmente identificable en cualquier caso.",
    },

    "expander_title": {
        "en": "What does this mean? (to share with the patient/physician)",
        "es": "¿Qué significa esto? (para compartir con el paciente/médico)",
    },
    "expander_body": {
        "en": "This model simulates the progressive loss of functional nephrons using two "
              "mechanisms: **hyperfiltration** (as nephrons are lost, the remaining ones "
              "become overloaded and are damaged faster) and **compensation** (eGFR stays "
              "stable while there is reserve, and drops faster near the end).\n\n"
              "The parameters are anchored to **published aggregate trial data** (the placebo "
              "arms of CREDENCE and EMPA-KIDNEY; DAPA-CKD is predicted out-of-sample). **It "
              "has not been validated on a prospective clinical cohort** — see "
              "`docs/MODEL_DOCUMENTATION.md` for the full project status.",
        "es": "Este modelo simula la pérdida progresiva de nefronas funcionales con dos "
              "mecanismos: **hiperfiltración** (al perderse nefronas, las restantes se "
              "sobrecargan y se dañan más rápido) y **compensación** (el eGFR se mantiene "
              "estable mientras hay reserva y cae más rápido cerca del final).\n\n"
              "Los parámetros están anclados a **datos agregados de ensayos publicados** (los "
              "brazos placebo de CREDENCE y EMPA-KIDNEY; DAPA-CKD se predice fuera de "
              "muestra). **No ha sido validado en una cohorte clínica prospectiva** — ver "
              "`docs/MODEL_DOCUMENTATION.md` para el estado completo.",
    },
    "expander_body_mimic": {
        "en": "This model simulates the progressive loss of functional nephrons using two "
              "mechanisms: **hyperfiltration** (as nephrons are lost, the remaining ones "
              "become overloaded and are damaged faster) and **compensation** (eGFR stays "
              "stable while there is reserve, and drops faster near the end).\n\n"
              "The ACTIVE parameters come from a **population mechanistic calibration on "
              "MIMIC-IV using robust nonlinear least squares, with patient-level bootstrap "
              "for parameter uncertainty** — not the trial-anchored default calibration. **It "
              "has not been validated on a prospective clinical cohort** — see the "
              "*Limitations* section of the README.",
        "es": "Este modelo simula la pérdida progresiva de nefronas funcionales con dos "
              "mecanismos: **hiperfiltración** (al perderse nefronas, las restantes se "
              "sobrecargan y se dañan más rápido) y **compensación** (el eGFR se mantiene "
              "estable mientras hay reserva y cae más rápido cerca del final).\n\n"
              "Los parámetros ACTIVOS provienen de una **calibración poblacional sobre "
              "MIMIC-IV** (mínimos cuadrados no lineales robustos, bootstrap por paciente), "
              "no de la calibración por defecto anclada a ensayos. **No ha sido validado en "
              "una cohorte clínica prospectiva** — ver la sección *Limitations* del README.",
    },
    "uacr_plot_title": {
        "en": "Predicted albuminuria (UACR) — a model OUTPUT, not an input",
        "es": "Albuminuria (UACR) predicha — una SALIDA del modelo, no una entrada",
    },
    "uacr_y": {"en": "predicted UACR (mg/g)", "es": "UACR predicha (mg/g)"},
    "uacr_note": {
        "en": "In this version albuminuria is **endogenous**: it rises as nephrons are lost "
              "(it is a readout of glomerular hypertension) and falls when renoprotective "
              "therapy lowers intraglomerular pressure. The model predicts a **{drop:.0f}%** "
              "immediate reduction on treatment; SGLT2i trials published 31–35%. This endpoint "
              "was structurally inexpressible in the previous model version.",
        "es": "En esta versión la albuminuria es **endógena**: sube al perderse nefronas (es un "
              "reflejo de la hipertensión glomerular) y baja cuando la terapia renoprotectora "
              "reduce la presión intraglomerular. El modelo predice una reducción inmediata de "
              "**{drop:.0f}%**; los ensayos de iSGLT2 publicaron 31–35%. Este desenlace era "
              "estructuralmente inexpresable en la versión anterior del modelo.",
    },
    "expander_body_v2": {
        "en": "This model simulates the progressive loss of functional nephrons using: "
              "**hyperfiltration** (as nephrons are lost, the survivors are overloaded and "
              "damaged faster — but with a *saturating*, physiologically bounded ceiling), "
              "**compensation** (eGFR holds while reserve remains, then falls faster), and "
              "**endogenous albuminuria** (UACR is a consequence of glomerular hypertension, "
              "not a fixed input).\n\n"
              "The active parameters are **anchored to published trial data**: progression is "
              "fixed by the placebo arms of CREDENCE and EMPA-KIDNEY, treatment effects by "
              "CREDENCE — and **DAPA-CKD is then predicted out-of-sample**, with both its "
              "chronic eGFR slope and its UACR reduction landing inside the published 95% CI.\n\n"
              "**It has still not been validated on a prospective clinical cohort**, and there "
              "is no acute haemodynamic dip term. See `the README (Limitations)`.",
        "es": "Este modelo simula la pérdida progresiva de nefronas mediante: "
              "**hiperfiltración** (al perderse nefronas, las restantes se sobrecargan y se "
              "dañan más rápido — pero con un techo *saturante*, acotado fisiológicamente), "
              "**compensación** (el eGFR se mantiene mientras hay reserva y luego cae más "
              "rápido) y **albuminuria endógena** (la UACR es consecuencia de la hipertensión "
              "glomerular, no una entrada fija).\n\n"
              "Los parámetros activos están **anclados a datos de ensayos publicados**: la "
              "progresión queda fijada por los brazos placebo de CREDENCE y EMPA-KIDNEY, y los "
              "efectos del tratamiento por CREDENCE — y luego **DAPA-CKD se predice fuera de "
              "muestra**, cayendo tanto su pendiente crónica como su reducción de UACR dentro "
              "del IC 95% publicado.\n\n"
              "**Sigue sin estar validado en una cohorte clínica prospectiva** y no incluye el "
              "dip hemodinámico agudo. Ver `the README (Limitations)`.",
    },
    "src_trial": {
        "en": "trial-anchored v2 (CREDENCE + EMPA-KIDNEY; DAPA-CKD predicted out-of-sample)",
        "es": "v2 anclada a ensayos (CREDENCE + EMPA-KIDNEY; DAPA-CKD predicho fuera de muestra)",
    },
    "training_estimator": {
        "en": "Preparing the personalization engine (one-off, ~15 s)…",
        "es": "Preparando el motor de personalización (una sola vez, ~15 s)…",
    },
    "history_header": {
        "en": "Past measurements (optional)",
        "es": "Mediciones previas (opcional)",
    },
    "history_caption": {
        "en": "Add earlier creatinine values to PERSONALIZE the model to this patient. "
              "Needs at least 3 measurements spanning at least 9 months.",
        "es": "Agrega creatininas anteriores para PERSONALIZAR el modelo a este paciente. "
              "Requiere al menos 3 mediciones que abarquen al menos 9 meses.",
    },
    "history_load_example": {
        "en": "Load an example history",
        "es": "Cargar un historial de ejemplo",
    },
    "history_example_notice": {
        "en": "Example measurement history — NOT patient data.",
        "es": "Historial de ejemplo — NO son datos de un paciente.",
    },
    "history_years_ago": {"en": "Years ago", "es": "Hace (años)"},
    "history_creat": {"en": "Creatinine (mg/dL)", "es": "Creatinina (mg/dL)"},
    "personalized_on": {
        "en": "**Personalized to this patient.** From their measurement history, the model "
              "infers an individual injury rate of **{scale:.2f}×** the population average "
              "(90% interval {scale_lo:.2f}–{scale_hi:.2f}). The projection below uses this, "
              "not the population default.\n\n"
              "*Experimental:* collapse exponent q = {q:.2f} (90% interval {q_lo:.2f}–"
              "{q_hi:.2f}). q is close to unidentifiable from routine data — treat it as "
              "indicative, not as an estimate.",
        "es": "**Personalizado a este paciente.** A partir de su historial, el modelo infiere "
              "una tasa de daño individual de **{scale:.2f}×** el promedio poblacional "
              "(intervalo 90% {scale_lo:.2f}–{scale_hi:.2f}). La proyección de abajo usa este "
              "valor, no el poblacional.\n\n"
              "*Experimental:* exponente de colapso q = {q:.2f} (intervalo 90% {q_lo:.2f}–"
              "{q_hi:.2f}). q es casi inidentificable con datos rutinarios — tómalo como "
              "indicativo, no como una estimación.",
    },
    "interval_note": {
        "en": "The 90% intervals are **conformalized**: the raw ensemble spread is badly "
              "over-confident (a naive 90% band covered the truth only ~33% of the time), so "
              "it is rescaled against held-out virtual patients until its coverage is "
              "actually 90%.",
        "es": "Los intervalos 90% están **conformalizados**: la dispersión cruda del ensamble "
              "es muy sobreconfiada (una banda 90% ingenua cubría la verdad sólo ~33% de las "
              "veces), así que se reescala contra pacientes virtuales retenidos hasta que su "
              "cobertura sea realmente del 90%.",
    },

    "personalized_off": {
        "en": "Using **population** parameters (anchored to published trials). To personalize, "
              "add at least 3 past creatinine measurements spanning at least 9 months in the "
              "sidebar.",
        "es": "Usando parámetros **poblacionales** (anclados a ensayos publicados). Para "
              "personalizar, agrega al menos 3 creatininas previas que abarquen al menos "
              "9 meses en la barra lateral.",
    },
    "personalized_caveat": {
        "en": "The individual injury rate is the reliable part of this personalization; **q is "
              "only weakly identifiable** from a handful of noisy measurements, so treat its "
              "value as indicative. See the README (Limitations).",
        "es": "La tasa de daño individual es la parte confiable de esta personalización; **q es "
              "sólo débilmente identificable** a partir de unas pocas mediciones ruidosas, así "
              "que su valor es indicativo. Ver el README (Limitations).",
    },
    "footer": {
        "en": "Source code and full documentation: "
              "[github.com/Danpc11/nephroq](https://github.com/Danpc11/nephroq)",
        "es": "Código fuente y documentación completa: "
              "[github.com/Danpc11/nephroq](https://github.com/Danpc11/nephroq)",
    },
    "src_public": {
        "en": "public (trial-anchored: CREDENCE + EMPA-KIDNEY)",
        "es": "pública (anclada a ensayos: CREDENCE + EMPA-KIDNEY)",
    },
}


def t(lang, key, **fmt):
    """Return the string for `key` in `lang` (falling back to English), with
    optional {placeholder} formatting."""
    entry = STRINGS.get(key, {})
    s = entry.get(lang) or entry.get("en") or key
    return s.format(**fmt) if fmt else s


# ------------------------------------------------------------------------------
# Example patients (clinical vignettes). Markers are language-independent;
# labels and "what to notice" notes are bilingual. Order matters for the demo
# (see docs/CLINICIAN_DEMO.md): start with the hidden-risk case.
# ------------------------------------------------------------------------------
PRESETS = [
    {
        "id": "fast",
        "label": {"en": "🔴 Fast progressor", "es": "🔴 Progresor rápido"},
        "markers": dict(age=58, sex="F", creatinine=1.3, hba1c=8.1, uacr=145.0, sbp=142),
        "note": {
            "en": "Already CKD G3a. The model shows an **accelerating, non-linear** decline "
                  "crossing the eGFR<15 threshold in a few years — and a visible shift in that "
                  "timeline when a renoprotective scenario is added. The 'act now' case.",
            "es": "Ya en ERC G3a. El modelo muestra un declive **acelerado y no lineal** que "
                  "cruza el umbral eGFR<15 en pocos años — y un cambio visible en ese plazo al "
                  "añadir un escenario renoprotector. El caso de 'actuar ahora'.",
        },
    },
    {
        "id": "hidden",
        "label": {"en": "🟠 Normal eGFR, hidden risk", "es": "🟠 eGFR normal, riesgo oculto"},
        "markers": dict(age=49, sex="M", creatinine=0.95, hba1c=9.2, uacr=280.0, sbp=150),
        "note": {
            "en": "eGFR looks **normal (G1)** on a single snapshot, but poor glycemic control + "
                  "high albuminuria drive hyperfiltration. The mechanistic model flags a future "
                  "progressor that a one-off eGFR would falsely reassure. **This is the case a "
                  "snapshot misses** — the core reason to model the mechanism, not just the number.",
            "es": "El eGFR se ve **normal (G1)** en una sola foto, pero el mal control glucémico + "
                  "la albuminuria alta impulsan la hiperfiltración. El modelo mecanístico señala un "
                  "futuro progresor que un eGFR aislado tranquilizaría falsamente. **Este es el caso "
                  "que una foto no ve** — la razón central de modelar el mecanismo, no solo el número.",
        },
    },
    {
        "id": "controlled",
        "label": {"en": "🟢 Well-controlled, low risk", "es": "🟢 Bien controlado, bajo riesgo"},
        "markers": dict(age=55, sex="F", creatinine=1.0, hba1c=6.6, uacr=15.0, sbp=125),
        "note": {
            "en": "Good control, low albuminuria: the model projects a **slow, near-flat** "
                  "trajectory. Shows the model does not cry wolf — a specificity / face-validity "
                  "check that matters as much as catching the progressors.",
            "es": "Buen control, albuminuria baja: el modelo proyecta una trayectoria **lenta y "
                  "casi plana**. Muestra que el modelo no exagera — una comprobación de "
                  "especificidad / validez aparente tan importante como detectar progresores.",
        },
    },
    {
        "id": "advanced",
        "label": {"en": "🟣 Advanced (G3b–G4)", "es": "🟣 Avanzado (G3b–G4)"},
        "markers": dict(age=63, sex="M", creatinine=2.1, hba1c=8.8, uacr=600.0, sbp=155),
        "note": {
            "en": "Already advanced. Illustrates the **terminal-collapse regime** where the "
                  "feedback exponent q dominates and small differences in control translate into "
                  "months, not years.",
            "es": "Ya avanzado. Ilustra el **régimen de colapso terminal** donde el exponente de "
                  "retroalimentación q domina y pequeñas diferencias de control se traducen en "
                  "meses, no años.",
        },
    },
]


def preset_by_id(pid):
    for p in PRESETS:
        if p["id"] == pid:
            return p
    return None
