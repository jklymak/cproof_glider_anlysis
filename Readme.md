---
title: Analysis for glider deployments
author: Jody Klymak
---

This is meant to be version controlled anlaysis set ups.  The html/pdf files
will be generated from these, but not version controlled.

To install an environment that works with these notebooks look at
`environment.yml` and for `conda env -f environment.yml`.  The only
requirement you may need to install manually is `pyglider`, which is
available via `pip install git+https://github.com/jklymak/pyglider.git`

We will keep the notebooks in python form using
Jupytext <https://jupytext.readthedocs.io/>.  This allows the notebooks to
be saved as plain text, and hence be under version control.   We are using the
``Pair notebook with light script`` option for Jupytext.  
