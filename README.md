# Risk Aware Belief-dependent Constrained POMDP Planning

This repository contains the code for the publication
> Andrey Zhitnikov, Vadim Indelman. Risk Aware Belief-dependent Constrained POMDP Planning

The first two authors contributed equally to this work.

The code was written and tested with Julia 1.8.0 (https://julialang.org/)

To run the experiments please use ```main.jl```

Note that to select either or not the scaling of CCSS is activated you shall set a boolean field ''scale'' in struct CCSS. True value means the scaling is activated.  

### Arguments



Argument | Description
---|---
--sol | The solver. Two possibilites exist CCSS, PCSS
--delta | delta of the constraint 
--epsilon | epsilon of the constaint. Active only if PCCS is selected in flag --sol
--obs | number of observation to expand from each belief at each depth of the belief tree
--L | the horizon
--simulations | Number of simulations (planning sessions) per repitition
--Nstatistics | Number of repititions
--num-particles | Number of the particles of the beief 
  


### Running example
``` 

julia  main.jl

``` 
