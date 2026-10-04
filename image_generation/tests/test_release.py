import json
import numpy as np
import pytest
import torch
from driftfm.tasks.sample import sample_checkpoint
from driftfm.tasks.smoke import run_smoke
from driftfm.training.checkpoints import load_checkpoint


def test_synthetic_train_sample_evaluate_and_dataset_free_reload(tmp_path):
    report_path = run_smoke(tmp_path / 'check')
    report = json.loads(report_path.read_text())
    assert report['metrics'] and all(np.isfinite(v) for v in report['metrics'].values())
    checkpoint = load_checkpoint(report['checkpoint'])
    assert checkpoint['step'] == 2
    for source in (tmp_path / 'check/latents').glob('*.npz'):
        source.unlink()
    result = sample_checkpoint(report['checkpoint'], output_dir=tmp_path / 'independent', seed=42, samples_per_class=2)
    original = np.load(tmp_path / 'check/samples_1/samples.npz')
    loaded = np.load(result)
    for name in original:
        np.testing.assert_array_equal(original[name], loaded[name])


def test_sampler_validates_sizes_before_accessing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        sample_checkpoint('missing.pt', output_dir=tmp_path, num_steps=0)


def test_training_refuses_to_overwrite_existing_checkpoint(tmp_path):
    from driftfm.training.trainer import train_from_config_path
    run_smoke(tmp_path / 'check')
    with pytest.raises(FileExistsError, match='already contains checkpoints'):
        train_from_config_path(tmp_path / 'check/smoke.yaml')


@pytest.mark.parametrize('dataset_name,num_classes,latent_dim', [('mnist_latent', 10, 6), ('ffhq_latent', 6, 8)])
def test_sampler_decodes_images_without_training_data(tmp_path, dataset_name, num_classes, latent_dim):
    from PIL import Image
    from driftfm.architectures import ConditionalTimeMLP, MnistConvAE, FFHQALAE, FFHQALAEConfig
    from driftfm.training.checkpoints import save_checkpoint
    params = dict(data_dim=latent_dim, hidden_sizes=[16, 16], num_classes=num_classes, conditioning_mode='class', class_embedding_dim=4)
    model = ConditionalTimeMLP(**params)
    checkpoint_path = tmp_path / 'generator.pt'
    save_checkpoint(checkpoint_path, {
        'model_state': model.state_dict(), 'ema_state': None,
        'resolved_architecture': {'name': 'mnist_latent_mlp' if num_classes == 10 else 'ffhq_latent_mlp', 'params': params},
        'dataset_info': {'data_dim': latent_dim, 'class_names': [str(i) for i in range(num_classes)]},
        'experiment': {'dataset': {'name': dataset_name}, 'method': {'name': 'drift_flow_matching', 'params': {'drift_form': 'split_v0'}}},
    })
    decoder_path = tmp_path / 'decoder.pt'
    if dataset_name == 'mnist_latent':
        decoder = MnistConvAE(latent_dim=latent_dim)
        save_checkpoint(decoder_path, {'latent_dim': latent_dim, 'model_state': decoder.state_dict()})
    else:
        decoder = FFHQALAE(config=FFHQALAEConfig(start_channel_count=2, max_channel_count=16, layer_count=2, latent_size=latent_dim, mapping_layers=2))
        decoder.save_project_checkpoint(decoder_path)
    result = sample_checkpoint(checkpoint_path, output_dir=tmp_path / 'sample', samples_per_class=2, decoder_checkpoint=decoder_path, batch_size=4)
    assert result.exists()
    with Image.open(result.parent / 'samples.png') as image:
        assert image.width > 0 and image.height > image.width
