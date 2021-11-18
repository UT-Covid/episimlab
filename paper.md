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

Computational models play a critical role in our scientific understanding of, and preparedness for, infectious diseases.
Such in silico disease models simulate real-world transmission dynamics, and are thereby well-suited for such tasks as early detection of novel pandemics, improving situational awareness during periods of high prevalence, and estimating efficacy of intervention strategies.
These models have proved valuable when responding to emergent epidemics such as the H1N1 flu pandemic, the Ebola epidemic, the Zika virus pandemic, and most recently, the COVID-19 pandemic (ref).
For example, compartmental disease models had BIG EFFECT IN EARLY STAGES OF COVID (refs).
As more data on case incidence, hospitalizations, and viral genomics became available, disease modelers were able to incorporate new data streams to improve the performance of these compartmental models.
These complex models are often expensive to develop, and few implementations of compartmental models shared a common software framework or application flow.
ALSO MAKES IT DIFFICULT TO REPRODUCE, DISTRIBUTE, AND COMPARE DIFFERENT MODELS.
Since EXPERTS EXPECT MORE PANDEMICS (ref), WE NEED TO SOLVE THIS PROBLEM.

Episimlab is a Python package that seeks to address this problem by providing a flexible framework for developing compartmental disease models.
Models in Episimlab are collections of modular components, known as `processes`, which can be added, removed, or replaced to modify the dynamics of the simulated disease.
In practice, this means that Episimlab supports development of models that:

1. Have many parameters, often with multiple, varying dimensions
2. Use any compartment structure that can be represented as a graph
3. Simulate interventions dynamically, such as administering vaccines only when case incidence exceeds a threshold.

The package is designed to be approachable; it includes several pre-packaged models that the user can run in a few lines of code.
Commonly used `processes` such as calculating the force of infection (FOI) ship with Episimlab, and can be modified using class inheritance in Python.
When the user chooses to add data streams or more complex transmission dynamics, they can easily do so by adding or replacing `processes` in the model.
Episimlab also provides a scalable and performant runtime for model execution, thanks to integration with packages in the PyData stack such as Dask [@dask], Xarray [@xarray], and `xarray-simlab` [@xsimlab].
Finally, Episimlab provides a standard for packaging, versioning, and sharing models, using Python's built-in class attributes.

# Statement of Need

- Developing epidemic models is time consuming and rarely compostable/reproducible (review ref?)
	- Subject matter experts such as epidemiologists often recapitulate routines that are common in compartmental epidemic models, such as calculating the force of infection.

`Episimlab` was designed in collaboration with Meyers (ref), and it's prototypes were
used in COVID stuff (refs). However, `Episimlab` is designed to be used by anyone developing compartmental disease models. The package is useful for students, since it provides a minimal, approachable boilerplate for developing basic models in pure Python. At the same time, it introduces and reinforces best practices in object-oriented software development, such as modularity and reproducibility. 

For disease modeling experts, `Episimlab` provides a platform that supports a wide variety of modeling use cases. 
It leverages concurrency in `xarray-simlab`, dataset chunking in `Dask`, and accelerated matrix math in `xarray`, so `Episimlab` models are performant even when using large (GB?) datasets. For example, Safegraph stuff (ref).

# References
