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
        "en": "**Demonstration mode** — projections are generated from a synthetic "
              "research calibration and must not be interpreted as individualized "
              "clinical predictions.",
        "es": "**Modo demostración** — las proyecciones provienen de una calibración "
              "sintética de investigación y no deben interpretarse como predicciones "
              "clínicas individualizadas.",
    },
    "quality_warning": {
        "en": "**Calibration quality warning** — the active MIMIC-IV calibration was "
              "flagged by calibrate_mimic.py as unreliable: {reasons}. Do not treat "
              "these projections as trustworthy. See docs/KNOWN_ISSUES.md.",
        "es": "**Aviso de calidad de calibración** — calibrate_mimic.py marcó la "
              "calibración MIMIC-IV activa como poco confiable: {reasons}. No trate "
              "estas proyecciones como confiables. Ver docs/KNOWN_ISSUES.md.",
    },
    # ---- sidebar -------------------------------------------------------------
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
              "(illustrative: SGLT2i/ACEi-ARB combined effect)",
        "es": "Ya recibe terapia renoprotectora "
              "(ilustrativo: efecto combinado iSGLT2/IECA-ARA)",
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
              "individual variability, or unknown future lab values (see docs/KNOWN_ISSUES.md).",
        "es": "Intervalo de incertidumbre de **parámetros** bootstrap al 90%, entre los "
              "remuestreos que cruzan el umbral: **{lo:.1f} – {hi:.1f} años**. Refleja "
              "solo la incertidumbre de los parámetros de calibración — no el ruido de "
              "medición, la variabilidad individual ni valores futuros desconocidos "
              "(ver docs/KNOWN_ISSUES.md).",
    },
    "boot_no_interval": {
        "en": "Fewer than half of the bootstrap resamples reach the threshold within this "
              "horizon, so no interval is shown here. Modeled time is best read as "
              "'>{horizon} years' for a majority of parameter resamples.",
        "es": "Menos de la mitad de los remuestreos bootstrap alcanzan el umbral en este "
              "horizonte, por lo que no se muestra intervalo. El tiempo modelado se lee "
              "mejor como '>{horizon} años' para la mayoría de los remuestreos.",
    },
    "boot_none": {
        "en": "No bootstrap parameter-uncertainty band available for this calibration — "
              "point estimate only (see docs/KNOWN_ISSUES.md).",
        "es": "No hay banda de incertidumbre bootstrap para esta calibración — solo "
              "estimación puntual (ver docs/KNOWN_ISSUES.md).",
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
        "en": "eGFR was calculated with creatinine only. Requesting cystatin C reduces "
              "the estimation error of the feedback exponent (q) by ~5×.",
        "es": "El eGFR se calculó solo con creatinina. Solicitar cistatina C reduce ~5× "
              "el error de estimación del exponente de retroalimentación (q).",
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
              "The model parameters were calibrated with hierarchical Bayesian inference on "
              "verified synthetic data and a first face-validity check against real "
              "published data. **It has not been validated on a prospective clinical "
              "cohort** — see `docs/MODEL_DOCUMENTATION.md` for the full project status.",
        "es": "Este modelo simula la pérdida progresiva de nefronas funcionales con dos "
              "mecanismos: **hiperfiltración** (al perderse nefronas, las restantes se "
              "sobrecargan y se dañan más rápido) y **compensación** (el eGFR se mantiene "
              "estable mientras hay reserva y cae más rápido cerca del final).\n\n"
              "Los parámetros se calibraron con inferencia bayesiana jerárquica sobre datos "
              "sintéticos verificados y una primera comprobación de validez aparente contra "
              "datos reales publicados. **No ha sido validado en una cohorte clínica "
              "prospectiva** — ver `docs/MODEL_DOCUMENTATION.md` para el estado completo.",
    },
    "footer": {
        "en": "Source code and full documentation: "
              "[github.com/Danpc11/nephroq](https://github.com/Danpc11/nephroq)",
        "es": "Código fuente y documentación completa: "
              "[github.com/Danpc11/nephroq](https://github.com/Danpc11/nephroq)",
    },
    "src_public": {
        "en": "public (synthetic + Al-Shamsi 2018 validation)",
        "es": "pública (sintética + validación Al-Shamsi 2018)",
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
