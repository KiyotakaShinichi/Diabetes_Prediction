"""Research code. Separate from the production serving path by design.

Nothing in this package is imported by app.py, streamlit_app.py or admin_app.py,
and nothing here writes to model_artifacts/ or provenance/. Research outputs go
to research_artifacts/, so a benchmark run can never be mistaken for - or
overwrite - a deployed model.
"""
