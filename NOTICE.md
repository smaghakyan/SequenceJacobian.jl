# Notice and attribution

## The toolbox and the methods

`SequenceJacobian.jl` translates the Python package **sequence-jacobian**
(<https://github.com/shade-econ/sequence-jacobian>) into Julia. The sequence-space
Jacobian method, the fake-news algorithm, the block architecture (simple, solved,
heterogeneous-agent, stage and combined blocks), the steady-state and transition
solvers, the estimation routines and the example models (RBC, Krusell-Smith,
one-asset HANK, two-asset HANK) were designed and written by

* Adrien Auclert
* Bence Bardóczy
* Matthew Rognlie
* Ludwig Straub

The intellectual authorship of the toolbox is theirs. This package only carries their
code into another language. Please cite their paper whenever you use it:

> Auclert, Adrien, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub. 2021. "Using the
> Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models."
> *Econometrica* 89 (5): 2375–2408. <https://doi.org/10.3982/ECTA17434>

sequence-jacobian is distributed under the MIT License. Its copyright and permission
notice is the one in
<https://github.com/shade-econ/sequence-jacobian/blob/master/LICENSE>, and it applies
to the translated portions of this package.

## Earlier Julia code

Parts of the source started from **SSJ.jl** by the Federal Reserve Bank of New York
DSGE team (<https://github.com/FRBNY-DSGE/SSJ.jl>), a Julia implementation of the same
toolbox. Five files started from it and say so in their header (`Source: FRBNY SSJ.jl`):
`src/utilities/interpolate.jl`, `src/utilities/solvers.jl`, `src/utilities/multidim.jl`,
`src/utilities/graph.jl` and `src/classes/jacobian_dict.jl`. Those portions are
distributed under SSJ.jl's BSD 3-clause license, reproduced here as its terms require.
This package is not endorsed by the Federal Reserve Bank of New York.

```
Copyright (c) Federal Reserve Bank of New York
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used
   to endorse or promote products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Validation tools

The validation tools in `src/utilities/diagnostics.jl` (brute-force Jacobians, the
fake-news check, steady-path and terminal-distribution checks, the channel
decomposition and MPC statistics) follow checks used in Yuxuan Zhao's sequence-space
notebooks, written here for arbitrary blocks.

## This port

The translation, the bug fixes listed in `CHANGELOG.md`, the extensions listed there
under "new in this port" (state-dependent and joint Markov transitions in stage blocks,
two-sided differencing of the whole household block, memory-saving stage blocks, the
`Continuous2D` stage-block Jacobian), the Julia-specific parts (output detection from
the function syntax tree, option handling), the `labor_choice_dag` test model and the
documentation are by Sargis Maghakyan.
Errors in the port are his, not the original authors'.
