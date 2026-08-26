"""Presentation layer for the two Streamlit entrypoints.

Scope boundary, deliberately narrow:

* this package renders - it never loads a model, scores a request, queries a
  database, or decides an A/B assignment;
* the entrypoints (``streamlit_app.py``, ``admin_app.py``) keep page
  orchestration: page config, loading artifacts, authentication, and the order
  sections appear in.

Four modules, no more. ``theme`` owns the small amount of CSS that Streamlit has
no primitive for, ``formatting`` owns value formatting shared by both apps, and
the two ``*_components`` modules own the sections each app is built from.
"""
