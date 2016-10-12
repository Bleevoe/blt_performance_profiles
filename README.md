# Symbolic elimination performance profiles
Code for reproducing performance profile of the symbolic elimination schemes for dynamic optimization proposed in the two (as of yet unpublished) references below. The code depends on JModelica.org. At the time of writing, the code is compatible with revision [9203](https://svn.jmodelica.org/trunk/?p=9203) of JModelica.org. Significantly older or newer revisions are unlikely to be compatible without modification.

Magnusson, F. (2016). *Numerical and Symbolic Methods for Dynamic Optimization.* Ph.D. thesis. Department of Automatic Control, Lund University, Sweden.

Magnusson, F. and J. Åkesson (2016). "Symbolic elimination in dynamic optimization based on block-triangular ordering". *Optimization Methods and Software.* Submitted for publication. 

The scripts are intended to allow for convenient (assuming JModelica.org has already been installed) reproduction of the exact same benchmark in the publications. Some settings, for example which problems to include in the benchmark or IPOPT settings, can quite easily be changed. Others, for example which schemes to use, are to some extent hardcoded and require some effort to change.
