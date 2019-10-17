# GraphIsomorphism_Torch

This is a reimplementation of [**How Powerful are Graph Neural Networks? ICLR 2019.**](https://arxiv.org/abs/1810.00826).

Here is the [official PyTorch implementation](https://github.com/weihua916/powerful-gnns). 

I rewrite the *load_data* function to handle [graph-kernel-datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) mentioned by  [yukiTakezawa:Graph Isomorphism Network](https://github.com/yukiTakezawa/GraphIsomorphismNetwork). The code of MLP and GIN has not much difference to the official.

Run model:
```
python main.py 
```