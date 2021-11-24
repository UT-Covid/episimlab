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

Computational models play a crucial role in our scientific understanding of, and preparedness for, infectious disease epidemics.
These in silico disease models simulate real-world transmission dynamics, and are thereby well-suited for such tasks as early detection of novel pandemics, improving situational awareness during periods of high prevalence, and estimating efficacy of intervention strategies.
These modeling approaches have proven valuable in responding to emergent epidemics such as the H1N1 flu pandemic, the Ebola epidemic, the Zika virus pandemic, and the recent COVID-19 pandemic (ref).
For example, during the early waves of the COVID-19 pandemic, compartmental disease models were instrumental for projecting case counts and hospitalizations [@Pierce2020report, @Pierce2020ieee].
As more data on case incidence, hospitalizations, and viral genomics became available, disease modelers are able to simulate increasingly complex disease dynamics.
Developing such complex compartmental models is time-consuming, however, and few implementations share a common software framework or application flow.
Therefore, there is a urgent need for more robust cyberinfrastructure in the field of epidemic modeling.

`Episimlab` is a software development kit (SDK) written in Python that seeks to address this problem by providing a flexible framework for developing compartmental disease models.
Models in Episimlab are collections of modular components, known as `processes`, which can be added, removed, or replaced to modify the dynamics of the simulated disease.
In practice, this means that Episimlab supports development of models that:

1. Have many parameters, often with multiple, varying dimensions
2. Use any compartment structure that can be represented as a graph
3. Simulate interventions dynamically, such as administering vaccines only when case incidence exceeds a threshold.

The package is designed to be approachable; it includes several pre-packaged models that the user can run in a few lines of code.
When the user chooses to add data streams or more complex transmission dynamics, they can easily do so by adding or replacing `processes` in the model.
In addition, Episimlab provides a scalable and performant runtime for model execution, thanks to integration with packages in the PyData stack such as Dask [@dask], Xarray [@xarray], and `xarray-simlab` [@xsimlab].
Finally, Episimlab provides a standard for packaging, versioning, and sharing models, using Python's built-in class attributes.

# Statement of Need

`Episimlab` is a Python package that accelerates development of complex compartmental disease models. 
It provides sufficient boilerplate such that the user can quickly instantiate a basic compartmental model such as an SIR model [@Kermack1927].
Simple models such as the SIR model are not unique to `Episimlab`; they have a long history of use in epidemiology since their origin in the early 20th century [@Ross1916, @Ross1917, @Ross1917pt3].
More recently, various implementations of compartmental models have been made publicly available and easily usable as open-source software packages (refs).
These packages are valuable because they simplify execution of many simple compartmental models, but their usage is limited to the discrete handful of model structures that are published with the package.
In addition, such projects rarely support complex models with more than 5 or 6 compartments, presumably because complex models are difficult to develop and reproduce.
While inspired by these previous works, `Episimlab` aims to support development of models with arbitrary complexity.
It gives the user flexibility to define key components of their compartmental model, such as the dimensionality of the Markov state matrix, the number of compartments, and custom stochastic behavior.

`Episimlab` was designed in collaboration with data scientists and epidemiologists from the UT Austin COVID-19 Modeling Consortium (ref). 
Specifically, prototypes of `Episimlab` were used in studies forecasting hospital burden due to the COVID-19 epidemic in Austin, Texas (refs). 
Although the package was originally designed with specific epidemiological use cases in mind, `Episimlab` is meant to be used by anyone developing compartmental disease models. 
The package is useful for students, since it provides a minimal, approachable boilerplate for developing basic models in pure Python. 
It introduces and reinforces best practices in object-oriented software development, such as modularity and reproducibility. 
For disease modeling experts, `Episimlab` provides a platform that supports a wide variety of modeling use cases. 
It leverages concurrency in `xarray-simlab`, dataset chunking in `Dask`, and accelerated matrix math in `xarray`, so `Episimlab` models are performant even when using large datasets.

## Unused Text

Unlike past compartmental modeling software, models in `Episimlab` are modular.
`Episimlab` models are essentially unique sets of discrete components.
These components, formally known as `processes` in the API, can be modified, exchanged, reused, and shared between models.
Commonly used `processes`, such as calculating the force of infection (FOI), ship with Episimlab, and can be modified using class inheritance in Python.

# References
