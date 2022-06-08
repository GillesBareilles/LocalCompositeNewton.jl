# LocalCompositeNewton.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GillesBareilles.github.io/LocalCompositeNewton.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GillesBareilles.github.io/LocalCompositeNewton.jl/dev)
[![Build Status](https://github.com/GillesBareilles/LocalCompositeNewton.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/GillesBareilles/LocalCompositeNewton.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/GillesBareilles/LocalCompositeNewton.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GillesBareilles/LocalCompositeNewton.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

*Setup* is as follow:
```julia
]update
]registry add https://github.com/GillesBareilles/OptimRegistry.jl
]add https://github.com/GillesBareilles/LocalCompositeNewton.jl
```

Experiments are executed with the commands:
```julia
using LocalCompositeNewton
LocalCompositeNewton.expe_maxquad()
LocalCompositeNewton.expe_eigmax()
```
