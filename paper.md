---
title: 'Episimlab: a Python package for modeling epidemics'
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
These _in silico_ disease models simulate real-world transmission dynamics and can be applied to tasks such as early detection of novel pandemics, improving situational awareness during periods of high prevalence, and estimating efficacy of intervention strategies.
These modeling approaches have proven valuable in responding to emergent epidemics such as the H1N1 flu pandemic, the Ebola epidemic, the Zika virus pandemic, and the recent COVID-19 pandemic [@Rivers2019].
For example, during the early waves of the COVID-19 pandemic, compartmental disease models were instrumental for projecting case counts and hospitalizations in Austin, Texas [@Pierce2020report; @Pierce2020ieee].
As more data on case incidence, hospitalizations, and pathogen genomics become available, disease modelers are able to simulate increasingly complex disease dynamics.
Developing such complex compartmental models is time-consuming.
In addition, relatively few model implementations share a common software framework or application flow, which complicates efforts to reproduce model outputs.
Therefore, there is a urgent need for more robust cyberinfrastructure in the field of epidemic modeling.

Episimlab is a software development kit (SDK) written in Python that seeks to address this problem by providing a flexible framework for developing compartmental disease models.
Models in Episimlab are collections of modular components, known as "processes", which can be added, removed, or replaced to modify the dynamics of the simulated disease.
Data is passed between processes, usually as N-dimensional arrays, using a standardized interface provided by the xarray-simlab [@xsimlab] package.
In practice, this means that Episimlab supports development of models that:

1. Implement any compartment structure that can be represented as a graph, including cyclic graphs
2. Have many input parameters with variable dimensionality, such as recovery rates that depend on age and risk factors, or contact patterns that depend on location.
3. Simulate interventions, including dynamic interventions such as administering vaccines only when case incidence exceeds a threshold
4. Incorporate one or more data sources that are too large to load eagerly into CPU memory.

The package is designed to provide a minimal but extensible boilerplate; it includes several pre-packaged models that the user can run in a few lines of code.
When the user implements different or more complex transmission dynamics, they can do so by adding or replacing processes in the model.
In addition, Episimlab provides a scalable and performant runtime environment for model execution, thanks to integration with packages in the PyData stack such as Dask [@dask], Xarray [@xarray], and xarray-simlab [@xsimlab].
Finally, Episimlab provides a standard for packaging, versioning, and sharing models, using Python's built-in class attributes.

# Statement of Need

Episimlab is a Python package that provides a common framework for rapid development of complex compartmental disease models. 
It provides sufficient boilerplate such that the user can quickly instantiate a basic compartmental model.
Basic models such as the SIR model - containing the three compartments susceptible (S), infected (I), and recovered (R) - are not unique to Episimlab; they have a long history of use ever since their origin in the early 20th century [@Ross1916; @Ross1917; @Ross1917pt3; @Kermack1927].
More recently, various implementations of compartmental models have been made publicly available and easily usable as open-source software packages [@Miller2019; @epydemic; @Jacob2021; @Gleamviz2011].
These packages are valuable because they simplify execution of many simple compartmental models, but their usage is limited to the discrete handful of model structures that are published with each package.
In addition, such projects rarely support complex models containing more than 5 or 6 compartments, in part because complex models are difficult to develop and reproduce.

Inspired by previous works, Episimlab aims to support development of models with arbitrary complexity.
It gives the user flexibility to define key components of their compartmental model, such as the dimensionality of the Markov state space [@Grimmett2020], the number of compartments, structure and rate of transitions between compartments, and custom stochastic behavior.
Therefore, Episimlab supports not only the handful of compartmental models that are included in the package, but also an indefinite number of compartmental model structures which can be tailored for specific modeling use cases.
This is accomplished by enforcing a modular paradigm of model development.
The package provides a library of lightweight Python classes, known as processes in the API, which comprise a model when combined with other processes.
The core process `ComptModel` is the only process shared by all Episimlab models. 
`ComptModel` implements a Gillespie algorithm that supports deterministic or stochastic discrete-time Markov chain models [@markov; @gillespie], using a generic model of compartmental disease.
Therefore, Episimlab does not support models that run in continuous time, are defined using differential equations, or are agent-based.
The pre-packaged models included in Episimlab draw transition matrix values from Poisson distributions, but model developers can easily replace the Poisson with other distribution functions.

Of note, comparable software such as `epydemic` [@epydemic] and GLEaMviz [@Gleamviz2011] also support generic models of compartmental disease. 
They do not, however, support arbitrary dimensionality in input variables or in the Markov state matrix, thereby limiting their usage to simulations that run in fixed-dimensional space.

Episimlab was originally designed with epidemiological use cases in mind via collaboration with data scientists and epidemiologists in the UT Austin COVID-19 Modeling Consortium. 
Specifically, prototypes of Episimlab were used in studies projecting hospital burden due to the COVID-19 epidemic in Austin, Texas [@Pierce2020report; @Pierce2020ieee]. 
Two of the pre-packaged models - `partition_v1` and `vaccine` - were developed by epidemiologists in the Consortium and migrated to Episimlab [@Yang2020; @Lachmann2021].

Although the package was originally intended for use by epidemiologists, it is useful for anyone developing compartmental models of disease spread. 
For students, it provides a minimal boilerplate for developing basic models in pure Python. 
It introduces and reinforces best practices in object-oriented software development such as modularity and reproducibility. 
For disease modeling experts, Episimlab provides a platform that supports a wide variety of modeling use cases. 
Simple models can be easily adapted into more complex ones, encouraging a model development approach that is rapid, iterative, and organic.
Under the hood, Episimlab leverages concurrency in xarray-simlab, dataset chunking in Dask, and accelerated matrix math in xarray, so models are performant even when using large input datasets.
The standardized structure of models and processes simplifies code sharing, therby promoting collaborative development within and between disease modeling teams.

# Dependencies

xarray xarray-simlab dask networkx matplotlib

# Related Packages

## epydemic

`epydemic` is a Python package that provides a common framework for building models of epidemic processes [@epydemic]. It supports simulations that are discrete-time synchronous or continuous-time stochastic (Gillespie). Like Episimlab, it supports a generic model for compartmental disease, allowing for flexibility in the compartmental model structure. In addition, it ships with several basic compartmental models such as SIR, SIS, and SEIR.

## EoN (Epidemics on Networks)

Epidemics on Networks (EoN) is a Python package that simulates disease dynamics for SIR and SIS models [@Miller2019]. The package includes numerical solutions for 20 different differential equation models, and supports complex contagions using the Gillespie algorithm [@gillespie].

## Eir

`Eir` is a Python package that simulates epidemics using compartmental models [@Jacob2021]. 
It includes 4 distinct models with different mobility dynamics. 
In additon, it provides utilities for inspecting transmission chains, analyzing state histories, and visualizing simulation results.

## GLEaMviz

The Global Epidemic and Mobility (GLEaMviz) framework is a software system for simulating spread of emergent diseases [@Gleamviz2011]. 
The framework is comprised of two major software components: the client-side graphical user interface (GUI) and the GLEaMviz simulation engine.
The simulation engine incorporates high-resolution demographic and mobility data and supports a generic model of compartmental disease, while the front-end GUI provides an intuitive interface for specifying the desired model structure.

# Acknowledgements

We would like to thank the UT Austin COVID-19 Modeling Consortium for their collaborative support. In particular, we acknowledge Dr. Lauren Ancel Meyers and her team for their guidance throughout the development process.

This work is supported by CDC Contract 75D-301-19-C-05930 and NIH Grant 3R01AI151176-01S1.

# References
