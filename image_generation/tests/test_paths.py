from pathlib import Path
import pytest
import yaml
from driftfm.config import load_experiment_config, coerce_experiment_config, dump_experiment_config
from driftfm.utils import normalize_path, resolve_runtime_config


def test_config_relative_roots_survive_other_cwd_and_checkpoint_reload(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    config = root / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'runtime': {'data_root': 'data', 'runs_root': 'out'},
        'dataset': {'name': 'mnist_latent', 'params': {'root_dir': '{data_root}/latents'}},
        'architecture': {'name': 'mnist_latent_mlp'}, 'method': {'name': 'drift_flow_matching'},
    }))
    monkeypatch.chdir(tmp_path)
    loaded = load_experiment_config(config)
    assert loaded.dataset.params['root_dir'] == str(root / 'data/latents')
    assert loaded.output.root_dir == str(root / 'out')
    monkeypatch.setenv('DRIFTFM_RUNS_ROOT', '/different/root')
    restored = coerce_experiment_config(loaded.to_dict())
    assert restored.to_dict() == loaded.to_dict()
    dump_experiment_config(loaded, tmp_path / 'saved.yaml')
    assert load_experiment_config(tmp_path / 'saved.yaml').to_dict() == loaded.to_dict()


def test_cli_relative_paths_use_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert normalize_path('checkpoints/model.pt') == tmp_path / 'checkpoints/model.pt'


def test_environment_defaults_and_explicit_root_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DRIFTFM_DATA_ROOT', str(tmp_path / 'external'))
    roots = resolve_runtime_config(None)
    assert roots['data_root'] == str(tmp_path / 'external')
    assert roots['mnist_root'] == str(tmp_path / 'external/mnist')
    assert resolve_runtime_config({'data_root': 'specific'})['data_root'] == str(tmp_path / 'specific')


def test_shipped_configs_resolve_to_module_storage():
    root = Path(__file__).resolve().parents[1]
    for name in ['mnist', 'ffhq']:
        loaded = load_experiment_config(root / 'configs/experiments' / f'{name}_driftfm.yaml')
        assert loaded.runtime.data_root == str(root / 'data')
        assert loaded.runtime.runs_root == str(root / 'runs')
        assert loaded.method.params['drift_form'] == 'split_v0'
        assert loaded.evaluation.num_sampling_steps == [1, 2, 5, 10, 20, 50]


def test_full_evaluation_config_relocates_checkpoint_dataset(tmp_path):
    from driftfm.evaluation.registry import _merge_evaluation_override
    old = coerce_experiment_config({
        'runtime': {'data_root': '/old/data'},
        'dataset': {'name': 'ffhq_latent', 'params': {'test_npz_path': '/old/data/test.npz'}},
        'architecture': {'name': 'ffhq_latent_mlp'}, 'method': {'name': 'drift_flow_matching'},
    })
    path = tmp_path / 'new.yaml'
    path.write_text(yaml.safe_dump({
        'runtime': {'data_root': './newdata'},
        'dataset': {'name': 'ffhq_latent', 'params': {'test_npz_path': '{data_root}/test.npz'}},
        'architecture': {'name': 'ffhq_latent_mlp'}, 'method': {'name': 'drift_flow_matching'},
    }))
    loaded = _merge_evaluation_override(old, path)
    assert loaded.dataset.params['test_npz_path'] == str(tmp_path / 'newdata/test.npz')


def test_evaluation_only_config_uses_authored_runtime_roots(tmp_path):
    from driftfm.evaluation.registry import _merge_evaluation_override
    old = coerce_experiment_config({
        'runtime': {'data_root': '/old/data', 'runs_root': '/old/runs'},
        'dataset': {'name': 'ffhq_latent'}, 'architecture': {'name': 'ffhq_latent_mlp'},
        'method': {'name': 'drift_flow_matching'},
    })
    path = tmp_path / 'eval.yaml'
    path.write_text(yaml.safe_dump({
        'runtime': {'data_root': './newdata'},
        'evaluation': {'params': {'real_npz': '{data_root}/test.npz'}},
    }))
    loaded = _merge_evaluation_override(old, path)
    assert loaded.evaluation.params['real_npz'] == str(tmp_path / 'newdata/test.npz')
    assert loaded.runtime.mnist_root == str(tmp_path / 'newdata/mnist')
    assert loaded.runtime.runs_root == '/old/runs'
