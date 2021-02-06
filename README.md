Grakn + Pytorch Geometric
=========================

Library for knowledge graph convolution with Grakn and Pytorch Geometric.

Basically trying to do what kglib does with tensorflow:

https://github.com/graknlabs/kglib

To see some example code using this library:

https://github.com/jorenretel/grakn-pytorch-example


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