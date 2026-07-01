# RL2gMCP
Reinforcement learning optimization of the graphical approach for multiple testing

## Local web app

Install the app dependencies:

```bash
pip install -r requirements-app.txt
```

Run the local Streamlit app:

```bash
streamlit run app.py
```

Optional TikZ graph rendering in the Optimized Graph tab uses a Tectonic binary.
The app looks for `tectonic` on `PATH` or at `.tools/tectonic/tectonic`.

The app runs on your local machine, and RL training / Monte Carlo evaluation use
your local CPU or GPU resources.
