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
CMD ["python", "system_twin.py", "--skip-slow", "--skip-bayes"]
