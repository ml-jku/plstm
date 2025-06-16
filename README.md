# pLSTM: parallelizable Linear Source Transition Mark Networks - Core
[![arXiv](https://img.shields.io/badge/arXiv-2506.11997-b31b1b.svg)](https://arxiv.org/abs/2506.11997)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Korbinian Pöppel<sup>**1,2**</sup>, Richard Freinschlag<sup>**1**</sup>, Thomas Schmied<sup>**1**</sup>, Wei Lin<sup>**1**</sup>,, Sepp Hochreiter<sup>**1,2**</sup>

<sup>**1**</sup>ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria\
<sup>**2**</sup>NXAI GmbH, Linz, Austria


This repository contains the [pLSTM](https://arxiv.org/abs/2506.11997) (parallelizable Linear Source Transition Mark networks) core implementations in `flax.nnx`, `flax.linen` and `torch`.
pLSTMs inherit ideas from Multi-Dimensional RNNs [Graves et al. 2007](http://arxiv.org/abs/0705.2011) and linear RNNs.
With the linearity, and changing the gating structure to the Source, Transition and Mark gates, we introduce a multi-dimensional parallel associative scan, on general directed acyclic graphs (DAGs) for parallelization.

![](./linearRNN_vs_pLSTM.svg)

pLSTMs also solve the vanishing/exploding gradient/activation problem on DAGs, similar to how the LSTM tackled them for RNNs on sequences.


## Configuration

All layers within pLSTM can be configured using the config classes in `plstm.config` composed by way of `compoconf` library.

## Framework Implementations

pLSTM offers implementations across multiple popular deep learning frameworks:
- `nnx`
- `linen`
- `torch`


## Graph Implementation

Please note that `plstm_graph` is currently only implemented in `torch`.


## References

MD-RNNs:
- Graves et al. 2007: Multi-Dimensional Recurrent Neural Networks [http://arxiv.org/abs/0705.2011](http://arxiv.org/abs/0705.2011)

linear RNNs (among lots of others):
- Schlag et al. 2021: Linear Transformers are Secretly Fast Weight Programmers [http://arxiv.org/abs/2102.11174](http://arxiv.org/abs/2102.11174)
- Orvieto et al. 2023: Resurrecting Recurrent Neural Networks for Long Sequences [http://arxiv.org/abs/2303.06349](http://arxiv.org/abs/2303.06349)
- Gu and Dao 2023: Mamba: Linear-time sequence modeling with selective state spaces [http://arxiv.org/abs/2312.00752](http://arxiv.org/abs/2312.00752)
- Yang et el. 2023: Gated Linear Attention Transformers with Hardware-Efficient Training [http://arxiv.org/abs/2312.06635](http://arxiv.org/abs/2312.06635)
- Beck et al. 2024: xLSTM: Extended Long Short Term Memory [http://arxiv.org/abs/2405.04517](http://arxiv.org/abs/2405.04517)

State-Tracking:
- Merrill et al. 2024: The Illusion of State in State Space Models [https://arxiv.org/abs/2404.08819](https://arxiv.org/abs/2404.08819)


## License

MIT License

## Citation

If you use this dataset in your research, please cite:

```bibtex

@misc{poppel_plstm_2025,
	title = {{pLSTM}: parallelizable {Linear} {Source} {Transition} {Mark} networks},
	shorttitle = {{pLSTM}},
	url = {http://arxiv.org/abs/2506.11997},
	doi = {10.48550/arXiv.2506.11997},
	urldate = {2025-06-16},
	publisher = {arXiv},
	author = {Pöppel, Korbinian and Freinschlag, Richard and Schmied, Thomas and Lin, Wei and Hochreiter, Sepp},
	month = jun,
	year = {2025},
	note = {arXiv:2506.11997 [cs]},
	keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
}

```
