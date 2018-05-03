# Classifying the Type of a Supernova from its Spectrum
## Final Project for Spring 2018 UC Berkeley Astro 250 Course
[![Build Status](https://travis-ci.org/benstahl92/python-ay250-homeworks.svg?branch=master)](https://travis-ci.org/benstahl92/python-ay250-homeworks) [![Coverage Status](https://coveralls.io/repos/github/benstahl92/python-ay250-homeworks/final_project/badge.svg?branch=master)](https://coveralls.io/github/benstahl92/python-ay250-homeworks?branch=master)

In this project I have built a codebase that provides end to end construction of trained machine learning classifiers for determining the type of a supernova (SN) from its spectrum. `fp.py` provides top level execution and makes calls to `Spectrum.py`, `ML_prep.py`, and `SNDB.py` for classes and functions that provide the meat of processing. Additionally, a (not-provided) `db_params.py` file is required and must specify login details for the database being used. Furthermore, the `SNDB.py` file should be modified according to the database being used.

The procedure followed by `fp.py` can be summarized as follows:
* retrieves spectral metadata (either from database query or from saved database query results)
* performs pre-processing steps on all spectra (or retrieves saved pre-processed data)
* featurizes pre-processed data for ingestion by ML models
* trains ML classifiers, computes success metrics, saves results to file

There is extensive documentation and examples/doctests in the docstrings for all functions and classes.
