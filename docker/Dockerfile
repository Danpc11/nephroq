# Reproducible environment for the system (TRL4 requirement: validation in a
# controlled laboratory environment, identical on any machine that runs it).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

# System sanity check at image build time: if the unit tests fail, the image
# does not build (quality gate).
RUN python -m pytest tests/ -v

WORKDIR /app/src
# Default run: the in-silico trial replication -- a fast, deterministic, self-
# contained validation that reproduces the three published placebo arms and
# writes results/insilico_trial_report.md. (The previous entrypoint,
# system_twin.py, was removed when the pipeline was restructured; its orchestrated
# stage scripts no longer exist, so pointing here keeps `docker run` working.)
CMD ["python", "insilico_trial.py"]
