TypeDB + Pytorch Geometric
==========================

:bangbang: Disclaimer:
These ideas have made it into **TypeDB ML** by now. Please Use that.
https://github.com/vaticle/typedb-ml
https://blog.vaticle.com/link-prediction-knowledge-graph-pytorch-geometric-f35917320806

Library for knowledge graph convolution with TypeDB and Pytorch Geometric.

Basically trying to do what kglib does with tensorflow:

https://github.com/graknlabs/kglib

To see some example code using this library:

https://github.com/bayer-science-for-a-better-life/grakn-pytorch-example


```
                                                                                     You are here
                                                                                          +
                                                                                          |
                                                                                          |
                                                                                          v
                                                                         +--------------------------------+
+------------------------+           +-------------------+               | +----------------------------+ |
|                        |           |                   |               | |                            | |
| Grakn Knowledge Graph  |   +---->  | grakn-dataloading |    +------->  | |  Deep Learning Framework X | |
|                        |           |                   |               | |                            | |
+------------------------+           +-------------------+               | +----------------------------+ |
                                                                         +--------------------------------+
```
