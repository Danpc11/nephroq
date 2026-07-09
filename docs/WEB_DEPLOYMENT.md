# How to bring the digital twin to a web interface

## The role of GitHub (and what it does NOT do on its own)

GitHub hosts the code and its version history. **It does not, by itself,
serve an interactive Python app** — for that you need a service that
*reads* your GitHub repo and *executes* the code on a server. GitHub Pages
only serves static sites (HTML/JS), and the twin needs to run Python
simulations, so Pages does not apply here.

## The chosen route: Streamlit

`app_web.py` (in the repo root) turns the model into an interactive web
app: the user enters a patient's markers and gets the eGFR projection, the
estimated time to dialysis, and the treated/untreated comparison — the same
logic as the Colab notebook, but as a real web page with a public URL.

### Run it locally

```bash
pip install -r requirements.txt
streamlit run app_web.py
```

Open `http://localhost:8501`.

### Deploy it for free, connected to GitHub

**Streamlit Community Cloud** (recommended, free for public repos):
1. Push the repo to GitHub (with `app_web.py` in the root).
2. Go to `share.streamlit.io`, connect your GitHub account.
3. Select the repo, the branch (`main`), and the file (`app_web.py`).
4. Deploy. You get a public URL (`your-app.streamlit.app`).
5. **Every `git push` to `main` redeploys automatically** — no manual step.

**Alternative: Hugging Face Spaces** — same flow (connect the GitHub repo or
upload directly), also free, with the advantage of integrating well with
models/datasets if the project grows in that direction.

Both options are free for academic/demo use and require no server
management.

## When to move to something more robust (FastAPI + React)

Streamlit is right for TRL4–5: demos, iteration with physicians, early
feedback. It's worth migrating to a **backend (FastAPI) + frontend
(React/Next.js)** architecture when any of these needs appear, typical of
TRL6+:

- Multiple users with authentication (physician/institution login).
- Storing patient histories in a database.
- Integration with an electronic health record (own API).
- Fine-grained interface control (Streamlit is more limited visually).
- Scaling to many concurrent users.

At that point, the mechanistic model (`src/mechanistic_twin.py`,
`egfr_measurement.py`, etc.) is reused unchanged — it just gets wrapped in
FastAPI endpoints instead of being called directly from `app_web.py`. The
already-existing `Dockerfile` is the natural base for that backend.

## Summary of the full flow

```
GitHub (code + version control)
   │
   ├── GitHub Actions (optional) → runs tests/ on every push
   │
   └── Streamlit Community Cloud / HF Spaces → reads the repo → deploys
       app_web.py as a public website, updated on every push
```

GitHub remains the center of everything; the web deployment is a layer
connected to it, not a replacement.
