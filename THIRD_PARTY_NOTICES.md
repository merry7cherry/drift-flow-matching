# Third-party notices

## ALAE

`image_generation/src/driftfm/architectures/ffhq_alae.py` adapts parts of the [official ALAE implementation](https://github.com/podgorskiy/ALAE), including the encoder/decoder and related layers, for inference within this package.

Copyright 2019–2020 Stanislav Pidhorskyi. The upstream source headers license these portions under the **Apache License, Version 2.0**. A copy is retained in [LICENSES/ALAE-Apache-2.0.txt](LICENSES/ALAE-Apache-2.0.txt). The local integration is modified from upstream. These portions remain under Apache 2.0; the root MIT license applies to original DFM code and documentation.

Reference: Pidhorskyi et al., *Adversarial Latent Autoencoders*, CVPR 2020.
