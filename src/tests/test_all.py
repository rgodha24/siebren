import numpy as np
import pytest
import torch
import torch.nn as nn

import siebren
from siebren import ReplayBuffer
from siebren.export import save_model, load_model_path, get_models_dir


class TestReplayBuffer:
    def test_create_buffer(self):
        buf = ReplayBuffer(1000, 10)
        assert len(buf) == 0
        assert buf.capacity == 1000
        assert buf.num_actions == 10

    def test_add_single(self):
        buf = ReplayBuffer(100, 10)
        states = {"board": np.random.randn(7, 32, 32).astype(np.float32)}
        policy = np.ones(10, dtype=np.float32) / 10
        value = 0.5

        buf.add(states, policy, value)
        assert len(buf) == 1

    def test_add_batch(self):
        buf = ReplayBuffer(100, 10)
        batch_size = 16
        states = {"board": np.random.randn(batch_size, 7, 32, 32).astype(np.float32)}
        policies = np.ones((batch_size, 10), dtype=np.float32) / 10
        values = np.zeros(batch_size, dtype=np.float32)

        buf.add_batch(states, policies, values)
        assert len(buf) == batch_size

    def test_sample(self):
        buf = ReplayBuffer(100, 10)
        # Add some data first
        for _ in range(20):
            states = {"board": np.random.randn(7, 32, 32).astype(np.float32)}
            policy = np.ones(10, dtype=np.float32) / 10
            buf.add(states, policy, 0.0)

        states, policies, values = buf.sample(8)

        assert "board" in states
        assert states["board"].shape == (8, 7, 32, 32)
        assert policies.shape == (8, 10)
        assert values.shape == (8, 1)

    def test_sample_returns_numpy(self):
        buf = ReplayBuffer(100, 5)
        buf.add(
            {"x": np.zeros((3, 3), dtype=np.float32)},
            np.ones(5, dtype=np.float32) / 5,
            0.0,
        )

        states, policies, values = buf.sample(1)

        assert isinstance(states["x"], np.ndarray)
        assert isinstance(policies, np.ndarray)
        assert isinstance(values, np.ndarray)

    def test_circular_buffer(self):
        capacity = 10
        buf = ReplayBuffer(capacity, 5)

        # Add more than capacity
        for i in range(25):
            buf.add(
                {"x": np.full((2, 2), i, dtype=np.float32)},
                np.ones(5, dtype=np.float32) / 5,
                float(i),
            )

        assert len(buf) == capacity

    def test_policy_size_mismatch_raises(self):
        buf = ReplayBuffer(100, 10)
        states = {"x": np.zeros((3, 3), dtype=np.float32)}
        wrong_policy = np.ones(5, dtype=np.float32)  # Wrong size

        with pytest.raises(ValueError):
            buf.add(states, wrong_policy, 0.0)

    def test_multiple_state_keys(self):
        buf = ReplayBuffer(100, 10)
        states = {
            "board": np.random.randn(7, 32, 32).astype(np.float32),
            "history": np.random.randn(5, 18).astype(np.float32),
        }
        policy = np.ones(10, dtype=np.float32) / 10

        buf.add(states, policy, 0.5)
        sampled_states, _, _ = buf.sample(1)

        assert "board" in sampled_states
        assert "history" in sampled_states
        assert sampled_states["board"].shape == (1, 7, 32, 32)
        assert sampled_states["history"].shape == (1, 5, 18)


class SimpleModel(nn.Module):
    """Simple model for testing export."""

    def forward(self, x: torch.Tensor):  # (B, 4)
        policy = x * 2
        value = x.sum(dim=1, keepdim=True)
        return policy, value


class TestExport:
    def test_save_model_creates_file(self, tmp_path):
        model = SimpleModel()
        path = save_model(
            model,
            name="test_simple",
            epoch=0,
            input_shapes={"x": (1, 4)},
            output_dir=tmp_path,
        )

        assert "test_simple_epoch_0000.onnx" in path
        assert (tmp_path / "test_simple" / "test_simple_epoch_0000.onnx").exists()

    def test_save_model_increments_epoch(self, tmp_path):
        model = SimpleModel()

        path0 = save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)
        path5 = save_model(model, "test", 5, {"x": (1, 4)}, tmp_path)
        path99 = save_model(model, "test", 99, {"x": (1, 4)}, tmp_path)

        assert "epoch_0000" in path0
        assert "epoch_0005" in path5
        assert "epoch_0099" in path99

    def test_load_model_path_specific_epoch(self, tmp_path):
        model = SimpleModel()
        save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)
        save_model(model, "test", 5, {"x": (1, 4)}, tmp_path)

        path = load_model_path("test", epoch=0, models_dir=tmp_path)
        assert "epoch_0000" in path

        path = load_model_path("test", epoch=5, models_dir=tmp_path)
        assert "epoch_0005" in path

    def test_load_model_path_latest(self, tmp_path):
        model = SimpleModel()
        save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)
        save_model(model, "test", 10, {"x": (1, 4)}, tmp_path)

        path = load_model_path("test", epoch=None, models_dir=tmp_path)
        assert "epoch_0010" in path

    def test_load_model_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_path("nonexistent", models_dir=tmp_path)

    def test_exported_model_valid_onnx(self, tmp_path):
        import onnx

        model = SimpleModel()
        path = save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)

        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

    def test_exported_model_inference(self, tmp_path):
        import onnxruntime as ort

        model = SimpleModel()
        path = save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)

        session = ort.InferenceSession(path)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        outputs = session.run(None, {"x": x})

        policy, value = outputs
        assert policy.shape == (1, 4)
        assert value.shape == (1, 1)

        # Check correctness
        np.testing.assert_allclose(policy, x * 2, rtol=1e-5)
        np.testing.assert_allclose(value, [[10.0]], rtol=1e-5)

    def test_exported_model_dynamic_batch(self, tmp_path):
        import onnxruntime as ort

        model = SimpleModel()
        path = save_model(model, "test", 0, {"x": (1, 4)}, tmp_path)

        session = ort.InferenceSession(path)

        # Test with batch size 1
        x1 = np.random.randn(1, 4).astype(np.float32)
        out1 = session.run(None, {"x": x1})
        assert out1[0].shape == (1, 4)

        # Test with batch size 8
        x8 = np.random.randn(8, 4).astype(np.float32)
        out8 = session.run(None, {"x": x8})
        assert out8[0].shape == (8, 4)


class TestLegacyFunction:
    def test_sum_as_string(self):
        assert siebren.sum_as_string(1, 1) == "2"
        assert siebren.sum_as_string(0, 0) == "0"
        assert siebren.sum_as_string(100, 200) == "300"
