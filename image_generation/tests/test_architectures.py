import pytest
import torch
from driftfm.architectures import ConditionalTimeMLP, MnistConvAE, create_architecture
from driftfm.config import ArchitectureSection
from driftfm.data import Conditioning
from driftfm.methods.drift_flow_matching import DriftFlowMatchingMethod


def test_conditional_mlp_requires_labels_and_preserves_shape():
    model = ConditionalTimeMLP(data_dim=6, hidden_sizes=[16, 16], num_classes=10, class_embedding_dim=4)
    x, t, h = torch.randn(3, 6), torch.rand(3, 1), torch.rand(3, 1)
    assert model(x, t, h, Conditioning(class_labels=torch.tensor([0, 1, 2]))).shape == x.shape
    with pytest.raises(ValueError, match='class_labels is required'):
        model(x, t, h)
    with pytest.raises(ValueError, match=r'must stay in \[0, 9\]'):
        DriftFlowMatchingMethod().sample(
            model, x, Conditioning(class_labels=torch.tensor([0, 1, 10])), num_steps=2,
        )


def test_guidance_checkpoint_architecture_rejected():
    with pytest.raises(ValueError, match='Unsupported legacy conditioning_mode'):
        create_architecture(ArchitectureSection(name='mnist_latent_mlp', params={
            'data_dim': 6, 'num_classes': 10, 'conditioning_mode': 'class_omega',
        }))


def test_mnist_autoencoder_shapes():
    model = MnistConvAE()
    x = torch.rand(4, 1, 28, 28)
    assert model.encode(x).shape == (4, 16)
    assert model.decode(model.encode(x)).shape == x.shape
    assert create_architecture(ArchitectureSection(name='mnist_conv_ae')).latent_dim == 16
