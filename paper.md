---
title: 'Episimlab: A Python package for modeling epidemics'
tags:
  - Python
  - epidemiology
  - modeling
  - markov
  - HPC
authors:
  - name: Ethan Ho 
    orcid: 0000-0002-7070-3321
    affiliation: 1
  - name: Kelly Pierce
    orcid: 0000-0002-6513-8305
    affiliation: 1
affiliations:
 - name: Texas Advanced Computing Center (TACC) - University of Texas at Austin
   index: 1
date: 16 November 2021
bibliography: paper.bib
---

# Summary

- Epidemic modeling is important
	- COVID (review ref)
	- But also cybersecurity (review ref)
- `Episimlab` is a Python package for development and execution of complex compartmental epidemic models. 
- Users can
	- Quickly get up and running with a model off the shelf
	- Add complexity to the model by adding or modifying modular components, known as `processes` in Episimlab
	- It provides a framework that enables users to develop models that are modular, extensible, reusable, and reproducible.
- Episimlab ships with performace-optimized implementations of commonly used disease modeling routines, such as the force of infection (FOI) calculation.

# Statement of need

- Developing epidemic models is time consuming and rarely compostable/reproducible (review ref?)
	- Subject matter experts such as epidemiologists often recapitulate routines that are common to compartmental epidemic models, such as calculating the force of infection.

`Episimlab` was designed in collaboration with Meyers (ref), and it's prototypes were
used in COVID stuff (refs). However, `Episimlab` is designed to be used by anyone developing compartmental disease models. The package is useful for students, since it provides a minimal, approachable boilerplate for developing basic models in pure Python. At the same time, it introduces and reinforces best practices in object-oriented software development, such as modularity and reproducibility. 

For disease modeling experts, `Episimlab` provides a platform that supports a wide variety of modeling use cases. 
It leverages concurrency in `xarray-simlab`, dataset chunking in `Dask`, and accelerated matrix math in `xarray`, so `Episimlab` models are performant even when using large (GB?) datasets. For example, Safegraph stuff (ref).

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

# References
