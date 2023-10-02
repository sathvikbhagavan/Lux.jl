---
title: 'Lux.jl: Bridging Scientific Computing & Deep Learning'
tags:
  - Julia
  - Deep Learning
  - Scientific Computing
  - Neural Ordinary Differential Equations
  - Deep Equilibrium Models
authors:
  - name: Avik Pal
    orcid: 0000-0002-3938-7375
    affiliation: "1"
affiliations:
 - name: Electrical Engineering and Computer Science, CSAIL, MIT
   index: 1
date: 2 October 2023
bibliography: paper.bib
---

# Summary

Combining Machine Learning and Scientific Computing have recently led to development of
methods like Universal Differential Equations, Neural Differential Equations, Deep Equilibrium Models, etc.,
which have been pushing the boundaries of physical sciences. However, every major deep learning
framework requires the numerical softwares to be rewritten to satisfy their specific requirements.
Lux.jl is a deep learning framework written in Julia with the correct abstractions to provide seamless
composability with scientific computing softwares. Lux uses pure functions to provide a
compiler and automatic differentiation friendly interface without compromising on the performance.

# Statement of Need

Julia already has quite a few well established Neural Network Frameworks –
Flux [@innes2018fashionable] and KNet [@yuret2016knet]. However, similar to Pytorch,
Tensorflow, etc. these frameworks were designed for typical Deep Learining workflows and
Scientific Computing workflows had to be tailored to fit into these frameworks.

Having to rewrite these workflows, which are often highly optimized, is a major barrier for
research in this domain.

## Switching Automatic Differentiation Frameworks

## Support for CPU, NVIDIA GPUs and AMD GPUs

## Composability with Scientific Computing Softwares

In this section, we will go over a couple of examples to show how Lux.jl can be used with
other scientific computing softwares. Lux.jl has an extensive
[manual](https://lux.csail.mit.edu/dev/manual/interface),
[tutorials](https://lux.csail.mit.edu/dev/tutorials/), and
[API Reference](https://lux.csail.mit.edu/dev/api/), showcasing the composability in more
details.

### Neural Ordinary Differential Equations

### Deep Equilibrium Models

Deep Equilibrium Models [@bai2019deep; @pal2023continuous] are a class of neural networks
where the output of the model is the steady state of a dynamical system defined by an
internal neural network.

## Ecosystem

# Limitations

Lux.jl is still in its early days of development and has the following known limitations:

* Training Small Neural Networks on CPUs are not optimized yet. For small networks,
  [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) [@simplechains] is the fastest option!
* Nested Automatic Differentiation is current not well supported. We hope to fix this soon,
  with a migration to Enzyme Automatic Differentiation Framework [@enzyme:2020; @enzyme:2021].

# References
