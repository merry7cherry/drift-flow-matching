from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flowviz.cli import parse_args
from flowviz.configs import (
    DATASET_CONFIGS, DriftFlowMatchingConfig, IntegratorConfig, MeanFlowConfig,
    TrainingConfig, VisualizationEvalConfig,
)
from flowviz.models.mlp import TwoTimeVelocityMLP
from flowviz.pipelines.checkpoints import load_checkpoint, save_checkpoint
from flowviz.pipelines.inference import compute_drift_flow_matching_trajectories
from flowviz.pipelines.training import (
    compute_drift_batched_sinkhorn, train_drift_flow_matching,
    train_flow_matching, train_mean_flow_matching,
)
from flowviz.runners.visualization import VisualizationRunConfig, run_visualization_experiments
from flowviz.seed import seed_all

torch.set_num_threads(1)
KEY = "2d_circular_uniform_to_moon"


class PublicWorkflowTest(unittest.TestCase):
    def test_drift_translation_and_group_independence(self):
        # Drift is a displacement and must be translation invariant. Each group
        # defines an independent empirical distribution and must not interact.
        torch.manual_seed(4)
        queries, targets = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        options = dict(temp_pos=.5, temp_neg=.5, sinkhorn_iters=10)
        drift = compute_drift_batched_sinkhorn(queries, targets, **options)
        shift = torch.tensor([3., -7.])
        torch.testing.assert_close(
            drift, compute_drift_batched_sinkhorn(queries + shift, targets + shift, **options),
            atol=1e-5, rtol=1e-5,
        )
        for group in range(2):
            torch.testing.assert_close(drift[group:group+1], compute_drift_batched_sinkhorn(
                queries[group:group+1], targets[group:group+1], **options,
            ))
        torch.testing.assert_close(compute_drift_batched_sinkhorn(queries, queries, **options), torch.zeros_like(queries))

    def test_all_methods_train_finite_and_checkpoint_round_trip(self):
        training = TrainingConfig(epochs=1, batch_size=16, steps_per_epoch=2)
        dataset = DATASET_CONFIGS[KEY].create_dataset(42)
        for name, train, config in [
            ("drift_flow_matching", train_drift_flow_matching, DriftFlowMatchingConfig()),
            ("flow_matching", train_flow_matching, None),
            ("mean_flow", train_mean_flow_matching, MeanFlowConfig()),
        ]:
            with self.subTest(method=name), TemporaryDirectory() as temp:
                seed_all(42)
                dataset.reset_rng(42)
                artifacts = train(dataset, training, config) if config else train(dataset, training)
                self.assertEqual(len(artifacts.history.losses), 1)
                self.assertTrue(np.isfinite(artifacts.history.losses).all())
                checkpoint = Path(temp)/"model.pt"
                save_checkpoint(checkpoint, artifacts.model, method=name, dataset_key=KEY,
                    hidden_sizes=(128,128,128), training_config=training,
                    method_config=asdict(config) if config else {}, seed=42, losses=artifacts.history.losses)
                loaded, metadata = load_checkpoint(checkpoint, torch.device("cpu"), expected_method=name, expected_dataset=KEY)
                self.assertEqual(metadata["training"]["steps_per_epoch"], 2)
                for actual, expected in zip(loaded.parameters(), artifacts.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_checkpoint(checkpoint, "cpu", expected_method=name, expected_dataset="wrong")
                if name == "drift_flow_matching":
                    points = dataset.sample_base(8, torch.device("cpu"))
                    original, _ = compute_drift_flow_matching_trajectories(artifacts.model, points, torch.device("cpu"), steps=3)
                    restored, _ = compute_drift_flow_matching_trajectories(loaded, points, torch.device("cpu"), steps=3)
                    torch.testing.assert_close(restored, original, rtol=0, atol=0)

    def test_saved_trace_reloads_without_training(self):
        with TemporaryDirectory() as temp:
            first, second = Path(temp)/"train", Path(temp)/"load"
            config = dict(seed=42, training_config=TrainingConfig(1,16,2),
                integrator_config=IntegratorConfig(2), drift_flow_config=DriftFlowMatchingConfig(),
                mean_flow_config=MeanFlowConfig(), eval_config=VisualizationEvalConfig((1,3), 8, 8))
            run_visualization_experiments([DATASET_CONFIGS[KEY]], VisualizationRunConfig(first, **config))
            with patch("flowviz.runners.visualization.train_drift_flow_matching", side_effect=AssertionError("must not retrain")):
                run_visualization_experiments([DATASET_CONFIGS[KEY]], VisualizationRunConfig(second, **config, load_dir=first/"checkpoints"))
            for nfe in (1,3):
                stem=f"{KEY}_drift_flow_matching_nfe_{nfe}"
                with np.load(first/f"{stem}.npz") as before, np.load(second/f"{stem}.npz") as after:
                    self.assertEqual(before["states"].shape, (nfe+1,8,2))
                    np.testing.assert_array_equal(before["states"], after["states"])
                    np.testing.assert_array_equal(before["target_samples"], after["target_samples"])
                self.assertGreater((second/f"{stem}.png").stat().st_size, 1000)
            manifest = json.loads((second/f"{KEY}_drift_flow_matching.json").read_text())
            self.assertEqual(manifest["checkpoint_metadata"]["training"]["epochs"], 1)
            self.assertEqual(len(manifest["checkpoint_sha256"]), 64)
            with np.load(first/f"{KEY}_drift_flow_matching_nfe_1.npz") as one, np.load(first/f"{KEY}_drift_flow_matching_nfe_3.npz") as three:
                np.testing.assert_array_equal(one["states"][0], three["states"][0])

    def test_singleton_groups_use_conditional_drift_and_train(self):
        query = torch.tensor([[[1., 2.]], [[-1., 3.]]])
        target = torch.tensor([[[4., 6.]], [[2., -2.]]])
        torch.testing.assert_close(compute_drift_batched_sinkhorn(query, target), target - query)
        self.assertEqual(parse_args(["--batch-size", "4"]).batch_size, 4)

        seed_all(41)
        initial = TwoTimeVelocityMLP(2).state_dict()
        seed_all(41)
        dataset = DATASET_CONFIGS[KEY].create_dataset(41)
        artifacts = train_drift_flow_matching(
            dataset, TrainingConfig(1, 4, 2), DriftFlowMatchingConfig(use_ema=False),
        )
        self.assertGreater(artifacts.history.losses[0], 0.)
        self.assertTrue(any(not torch.equal(initial[key], value)
                            for key, value in artifacts.model.state_dict().items()))

    def test_comparison_uses_shared_bounds_for_all_methods_and_nfe(self):
        from flowviz.visualization.plotting import create_2d_trajectory_figure

        with TemporaryDirectory() as temp:
            output = Path(temp)
            config = VisualizationRunConfig(
                output, seed=42, training_config=TrainingConfig(1, 16, 2),
                integrator_config=IntegratorConfig(2), drift_flow_config=DriftFlowMatchingConfig(),
                mean_flow_config=MeanFlowConfig(), eval_config=VisualizationEvalConfig((1, 3), 8, 8),
                methods=("drift_flow_matching", "flow_matching", "mean_flow"),
            )
            with patch("flowviz.runners.visualization.create_2d_trajectory_figure",
                       wraps=create_2d_trajectory_figure) as plot:
                run_visualization_experiments([DATASET_CONFIGS[KEY]], config)
            self.assertEqual(plot.call_count, 5)
            bounds = plot.call_args_list[0].kwargs["bounds"]
            points = []
            sources = []
            for path in output.glob("*.npz"):
                with np.load(path) as trace:
                    points.extend((trace["states"].reshape(-1, 2), trace["target_samples"]))
                    sources.append(trace["states"][0])
            all_points = np.concatenate(points)
            lower, upper = all_points.min(axis=0), all_points.max(axis=0)
            padding = np.maximum(upper - lower, 1.0) * 0.05
            np.testing.assert_array_equal(bounds, (lower - padding, upper + padding))
            for call in plot.call_args_list:
                np.testing.assert_array_equal(call.kwargs["bounds"], bounds)
            for source in sources[1:]:
                np.testing.assert_array_equal(source, sources[0])
            for path in output.glob("*.json"):
                np.testing.assert_array_equal(json.loads(path.read_text())["plot"]["bounds"], bounds)

    def test_cli_defaults_and_json_overrides(self):
        args=parse_args([])
        self.assertEqual(args.methods,["drift_flow_matching"])
        self.assertEqual(args.datasets,[KEY])
        self.assertEqual(args.device,"cpu")
        with TemporaryDirectory() as temp:
            path=Path(temp)/"config.json"
            path.write_text(json.dumps({"epochs":5,"nfe":[1,7],"output":"elsewhere"}))
            args=parse_args(["--config",str(path),"--epochs","2"])
            self.assertEqual(args.epochs,2)
            self.assertEqual(args.nfe,[1,7])
            self.assertEqual(args.output,Path("elsewhere"))
            path.write_text('{"epochs": 0}')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config",str(path)])

    def test_cli_rejects_invalid_budgets_and_unsupported_extensions(self):
        for args in (["--batch-size","7"],["--nfe","0"],["--epochs","0"],["--lr","nan"],
                     ["--methods","improved_mean_flow"],["--methods","drift_flow_matching[use_self_consistency_loss=True]"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)


if __name__ == "__main__":
    unittest.main()
