#!/usr/bin/env python3
"""Regression tests for Eagle draft RoPE configuration (issue #4895).

Uses tiny random weights, without downloading model checkpoints. Requires the
LLM export dependencies and onnx. Run from the repository root:
    python tools/script/testEagleExport.py

To also round-trip a checkpoint and run Eagle.export(), pass --output-dir with
an existing directory. Requires the MNN Python package; artifacts are retained
in a unique subdirectory. Add --converter to verify the MNN conversion too:
    python tools/script/testEagleExport.py --output-dir build_mnn --converter build_mnn/MNNConvert
Alternatively, --python-converter uses the installed MNN converter extension,
which may differ from the current checkout. Both modes compare the unfused,
FP32 graph with explicit KV inputs/outputs and external weights. Stateful
Attention fusion and weight quantization require separate runtime tests.
"""

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from transformers import PretrainedConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformers/llm/export"))
from utils.config import LlmConfig
from utils.eagle import LlamaEagle
from utils.model_mapper import ModelMapper
from utils.torch_utils import onnx_export
from utils.transformers import Rotary


class ReferenceRotary(torch.nn.Module):
    """Independent, direct-angle reference for ordinary Llama RoPE."""

    def __init__(self, head_dim, theta, ratio):
        super().__init__()
        base = theta * (1.0 if ratio is None else ratio)
        self.inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    def forward(self, position_ids):
        angles = position_ids.reshape(-1, 1).double() * self.inv_freq.double()
        cos = angles.cos().float()
        sin = angles.sin().float()
        return torch.cat((torch.stack((cos, sin)), torch.stack((cos, sin))), dim=-1).unsqueeze(1).unsqueeze(3)


class EagleExportTest(unittest.TestCase):
    output_dir = None
    converter = None

    def make_eagle(self, **overrides):
        torch.manual_seed(4895)
        config = LlmConfig(
            model_type="llama", hidden_size=8, num_attention_heads=2,
            num_key_value_heads=1, num_hidden_layers=8, head_dim=4,
            layer_types=["full_attention"] * 8,
            rope_theta=500000.0, rope_ratio=2.0,
        )
        config.model_map = ModelMapper().get_map(config)[1]
        rotary = Rotary(config)
        config.rotary = rotary
        base = SimpleNamespace(
            config=config, rotary=rotary,
            embed=SimpleNamespace(embed=torch.nn.Embedding(16, 8)),
        )
        fields = dict(
            hidden_size=8, vocab_size=16, draft_vocab_size=12,
            intermediate_size=16, rms_norm_eps=1e-6, hidden_act="silu",
        )
        fields.update(overrides)
        draft_config = PretrainedConfig(**fields)
        # Only checkpoint I/O is mocked; construction, attention and export are real.
        with patch("utils.eagle.PretrainedConfig.from_json_file", return_value=draft_config), \
                patch.object(LlamaEagle, "load"):
            eagle = LlamaEagle("unused-eagle-checkpoint", base).eval()
        return eagle, base

    def inputs(self, eagle, seq_len=3, history_len=0, start_pos=0, logits_index=-1):
        hidden = eagle.hidden_size
        kv_heads = eagle.num_key_value_heads
        head_dim = eagle.head_dim
        mask = torch.zeros(1, 1, seq_len, history_len + seq_len)
        mask[..., history_len:] = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(torch.float32).min), diagonal=1)
        return (
            torch.randn(seq_len, hidden), torch.randn(1, seq_len, hidden), mask,
            torch.arange(start_pos, start_pos + seq_len, dtype=torch.int32).unsqueeze(0),
            torch.randn(2, 1, history_len, kv_heads, head_dim),
            torch.tensor([logits_index], dtype=torch.int32),
        )

    def test_legacy_defaults(self):
        eagle, _ = self.make_eagle()
        self.assertEqual(eagle.rope_theta, 10000)
        self.assertEqual(eagle.rope_ratio, 1.0)
        expected = ReferenceRotary(eagle.head_dim, 10000, 1.0)
        positions = torch.tensor([[0, 1, 2047, 2048, 8193]], dtype=torch.int32)
        torch.testing.assert_close(eagle.config.rotary(positions), expected(positions), rtol=1e-6, atol=2e-7)

    def test_configured_parameters(self):
        for overrides, theta, ratio in [
            ({"rope_theta": 1000000.0}, 1000000.0, 1.0),
            ({"rope_ratio": 4.0}, 10000.0, 4.0),
            ({"rope_theta": 1000000.0, "rope_ratio": 4.0}, 1000000.0, 4.0),
            ({"rope_theta": 1000000.0, "rope_ratio": None}, 1000000.0, None),
        ]:
            with self.subTest(overrides=overrides):
                eagle, _ = self.make_eagle(**overrides)
                self.assertEqual(eagle.rope_theta, theta)
                self.assertEqual(eagle.rope_ratio, ratio)
                positions = torch.tensor([[0, 17, 2047, 2048, 8193, 131071]], dtype=torch.int32)
                expected = ReferenceRotary(eagle.head_dim, theta, ratio)(positions)
                torch.testing.assert_close(eagle.config.rotary(positions), expected, rtol=1e-6, atol=2e-7)

    def test_target_configuration_is_unchanged(self):
        for head_dim in (2, 4, 8):
            with self.subTest(head_dim=head_dim):
                eagle, base = self.make_eagle(head_dim=head_dim, rope_theta=1000000.0, rope_ratio=4.0)
                self.assertEqual(base.config.head_dim, 4)
                self.assertEqual(base.config.rope_theta, 500000.0)
                self.assertEqual(base.config.rope_ratio, 2.0)
                self.assertIs(base.config.rotary, base.rotary)
                self.assertIsNot(eagle.config, base.config)
                self.assertEqual(eagle.head_dim, head_dim)
                self.assertIs(eagle.embed_tokens.weight, base.embed.embed.weight)

    def test_attention_uses_draft_rotary(self):
        for head_dim in (2, 4, 8):
            with self.subTest(head_dim=head_dim):
                eagle, base = self.make_eagle(head_dim=head_dim)
                self.assertIs(eagle.midlayer.self_attn.rotary, eagle.config.rotary)
                self.assertIsNot(eagle.midlayer.self_attn.rotary, base.rotary)
                self.assertEqual(eagle.midlayer.self_attn.rope_cut_head_dim, head_dim)

    def test_forward_matches_reference_rope(self):
        for overrides, theta, ratio in [({}, 10000, 1.0),
                                         ({"rope_theta": 1000000.0, "rope_ratio": 4.0}, 1000000.0, 4.0)]:
            for head_dim in (2, 4, 8):
                with self.subTest(overrides=overrides, head_dim=head_dim):
                    eagle, _ = self.make_eagle(head_dim=head_dim, **overrides)
                    inputs = self.inputs(eagle, seq_len=5, history_len=2, start_pos=8191, logits_index=0)
                    with torch.no_grad():
                        actual = eagle(*inputs)
                        with patch.object(eagle.config, "rotary", ReferenceRotary(head_dim, theta, ratio)):
                            expected = eagle(*inputs)
                    for output, reference in zip(actual, expected):
                        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)

    def test_onnx_export_matches_reference(self):
        eagle, _ = self.make_eagle(head_dim=8, rope_theta=1000000.0, rope_ratio=4.0)
        input_names = ["input_embed", "hidden_states", "attention_mask", "position_ids",
                       "past_key_values", "logits_index"]
        buffer = io.BytesIO()
        with torch.no_grad():
            onnx_export(
                eagle, self.inputs(eagle), buffer, input_names,
                ["logits", "out_hidden_states", "presents"],
                dynamic_axes={
                    "input_embed": {0: "seq_len"}, "hidden_states": {1: "seq_len"},
                    "attention_mask": {2: "seq_len", 3: "kv_seq_len"},
                    "position_ids": {1: "seq_len"}, "past_key_values": {2: "history_len"},
                },
            )
        model = onnx.load_model_from_string(buffer.getvalue())
        onnx.checker.check_model(model)
        evaluator = ReferenceEvaluator(model)
        for seq_len, history_len, start_pos in [(3, 0, 0), (5, 2, 2047), (1, 7, 8193)]:
            with self.subTest(seq_len=seq_len, history_len=history_len, start_pos=start_pos):
                inputs = self.inputs(eagle, seq_len, history_len, start_pos)
                actual = evaluator.run(None, {name: value.numpy() for name, value in zip(input_names, inputs)})
                with torch.no_grad(), patch.object(eagle.config, "rotary", ReferenceRotary(8, 1000000.0, 4.0)):
                    expected = eagle(*inputs)
                for output, reference in zip(actual, expected):
                    np.testing.assert_allclose(output, reference.numpy(), rtol=1e-5, atol=1e-6)

    def check_mnn_conversion(self, model_path, case_dir, inputs, outputs):
        case_dir.mkdir()
        (case_dir / "input.json").write_text(json.dumps({
            "inputs": [{"name": name, "shape": list(value.shape)} for name, value in inputs.items()],
            "outputs": list(outputs),
        }))
        for name, value in {**inputs, **outputs}.items():
            np.savetxt(case_dir / (name + ".txt"), value.detach().numpy().reshape(-1), fmt="%.9g")
        backend = case_dir / "cpu.json"
        backend.write_text(json.dumps({"backend": 0, "precision": 1, "memory": 0}))
        # Attention fusion owns its KV cache and no longer returns the explicit
        # presents tensor. Compare the raw graph here, including every KV value.
        result = subprocess.run(
            [*self.converter, "-f", "ONNX", "--modelFile", str(model_path),
             "--MNNModel", str(case_dir / "model.mnn"), "--keepInputFormat=1",
             "--transformerFuseC4=0", "--saveExternalData",
             "--testdir", str(case_dir), "--testconfig", str(backend), "--thredhold", "1e-5"],
            cwd=str(case_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=120,
        )
        (case_dir / "convert.log").write_text(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Converted Success!", result.stdout)
        self.assertIn("TEST_SUCCESS", result.stdout)
        self.assertNotIn("TESTERROR", result.stdout)
        self.assertNotIn("skip", result.stdout.lower())
        for name, expected in outputs.items():
            match = re.search(re.escape(name) + r": \(([^)]*)\)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            shape = tuple(int(dim.strip()) for dim in match.group(1).split(",") if dim.strip())
            self.assertEqual(shape, tuple(expected.shape), result.stdout)

    def test_checkpoint_export_entry(self):
        if self.output_dir is None:
            self.skipTest("pass --output-dir to run checkpoint and Eagle.export() integration")
        import MNN.expr as expr
        workdir = Path(tempfile.mkdtemp(prefix="eagle4895-", dir=self.output_dir))
        print("\nEagle export artifacts:", workdir)
        input_names = ["input_embed", "hidden_states", "attention_mask", "position_ids",
                       "past_key_values", "logits_index"]
        output_names = ["logits", "out_hidden_states", "presents"]
        variants = [("legacy", {}, 10000.0, 1.0),
                    ("configured", {"head_dim": 8, "rope_theta": 1000000.0, "rope_ratio": 4.0},
                     1000000.0, 4.0)]
        for name, overrides, theta, ratio in variants:
            with self.subTest(variant=name):
                folder = workdir / name
                folder.mkdir()
                onnx_dir = folder / "onnx"
                onnx_dir.mkdir()
                original, base = self.make_eagle(**overrides)
                original.d2t.copy_(torch.arange(original.draft_vocab_size) % 4)
                (folder / "config.json").write_text(original.eagle_config.to_json_string())
                torch.save(original.state_dict(), folder / "pytorch_model.bin")
                # No mocked I/O here: read the serialized config and weights.
                eagle = LlamaEagle(str(folder), base).eval()
                self.assertEqual(eagle.rope_theta, theta)
                self.assertEqual(eagle.rope_ratio, ratio)
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(eagle.state_dict()[key], value, rtol=0, atol=0)
                model_path, fc_path = map(Path, eagle.export(str(onnx_dir)))
                for path in (model_path, fc_path, folder / "eagle_d2t.mnn"):
                    self.assertGreater(path.stat().st_size, 0)
                # read() returns a view, so retain the MNN variable while comparing.
                loaded_d2t = expr.load_as_dict(str(folder / "eagle_d2t.mnn"))["d2t"]
                np.testing.assert_array_equal(loaded_d2t.read(), eagle.d2t.numpy())
                model, fc_model = onnx.load(str(model_path)), onnx.load(str(fc_path))
                onnx.checker.check_model(model)
                onnx.checker.check_model(fc_model)
                evaluator, fc_evaluator = ReferenceEvaluator(model), ReferenceEvaluator(fc_model)
                cases = [(3, 0, 0, 0), (5, 2, 2047, 0), (1, 7, 8193, -1)]
                for seq_len, history_len, start_pos, logits_index in cases:
                    with self.subTest(variant=name, seq_len=seq_len, history_len=history_len):
                        inputs = self.inputs(eagle, seq_len, history_len, start_pos, logits_index)
                        feed = dict(zip(input_names, inputs))
                        actual = evaluator.run(None, {key: value.numpy() for key, value in feed.items()})
                        with torch.no_grad(), patch.object(
                                eagle.config, "rotary", ReferenceRotary(eagle.head_dim, theta, ratio)):
                            expected = eagle(*inputs)
                        self.assertEqual(len(actual), len(expected))
                        for output, reference in zip(actual, expected):
                            self.assertEqual(output.shape, tuple(reference.shape))
                            np.testing.assert_allclose(output, reference.numpy(), rtol=1e-5, atol=1e-6)
                        fc_input = torch.randn(1, seq_len, eagle.fc.in_features)
                        with torch.no_grad():
                            fc_expected = eagle.fc(fc_input)
                        fc_actual = fc_evaluator.run(None, {"fc_hidden": fc_input.numpy()})[0]
                        np.testing.assert_allclose(fc_actual, fc_expected.numpy(), rtol=1e-5, atol=1e-6)
                        if self.converter:
                            tag = "case_{}_{}_{}".format(seq_len, history_len, start_pos)
                            self.check_mnn_conversion(model_path, folder / tag, feed,
                                                      dict(zip(output_names, expected)))
                            self.check_mnn_conversion(fc_path, folder / (tag + "_fc"),
                                                      {"fc_hidden": fc_input}, {"hidden_states": fc_expected})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output-dir", type=Path)
    converter_group = parser.add_mutually_exclusive_group()
    converter_group.add_argument("--converter", type=Path)
    converter_group.add_argument("--python-converter", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.output_dir is not None:
        if not args.output_dir.is_dir():
            parser.error("--output-dir must be an existing directory")
        EagleExportTest.output_dir = str(args.output_dir.resolve())
    if args.converter is not None:
        if args.output_dir is None or not args.converter.is_file():
            parser.error("--converter requires --output-dir and an existing converter executable")
        EagleExportTest.converter = [str(args.converter.resolve())]
    elif args.python_converter:
        if args.output_dir is None:
            parser.error("--python-converter requires --output-dir")
        EagleExportTest.converter = [
            sys.executable, "-B", "-c",
            "import sys, _tools; _tools.mnnconvert(['MNNConvert', *sys.argv[1:]])",
        ]
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
