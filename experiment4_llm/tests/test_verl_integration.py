"""Tests for verl integration (custom estimators in ray_trainer.py)."""
import torch
import numpy as np
import pytest
import inspect


def test_custom_estimators_in_enum():
    """Custom estimator names exist in AdvantageEstimator enum."""
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        assert hasattr(AdvantageEstimator, name.upper()), \
            f"{name.upper()} not in AdvantageEstimator enum"
        member = getattr(AdvantageEstimator, name.upper())
        assert member.value == name


def test_custom_estimators_registered():
    """All custom estimators are in CUSTOM_ESTIMATORS."""
    import experiment4_llm.src.verl_register as reg

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        assert name in reg.CUSTOM_ESTIMATORS, f"{name} not registered"


def test_custom_estimator_signature():
    """Each custom fn returns (advantages, returns) tuple with correct shapes."""
    import experiment4_llm.src.verl_register as reg

    B, N, seq_len = 4, 4, 10
    token_rewards = torch.zeros(B * N, seq_len)
    mask = torch.ones(B * N, seq_len)
    index = np.repeat(np.arange(B), N)
    # Place random rewards at last token
    for i in range(B * N):
        token_rewards[i, -1] = torch.rand(1).item()

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        fn = reg.CUSTOM_ESTIMATORS[name]
        result = fn(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )
        assert isinstance(result, tuple), f"{name} didn't return tuple"
        assert len(result) == 2, f"{name} returned {len(result)} elements"
        adv, ret = result
        assert adv.shape == (B * N, seq_len)
        assert ret.shape == (B * N, seq_len)


def test_compute_advantage_dispatches():
    """compute_advantage in ray_trainer dispatches to our custom estimators."""
    from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator

    # This test verifies the dispatch path exists (doesn't raise NotImplementedError)
    # We can't easily call compute_advantage without a full DataProto,
    # but we can verify the enum values are accepted
    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        member = AdvantageEstimator(name)
        assert member.value == name


def test_compute_advantage_full_dispatch():
    """Test compute_advantage() dispatches to custom estimators with real DataProto.

    Constructs a minimal DataProto with token_level_rewards, response_mask,
    and uid, then verifies compute_advantage() runs end-to-end without error
    and produces correctly-shaped, non-NaN outputs.
    """
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
    from tensordict import TensorDict

    B, N, seq_len = 4, 4, 10
    total = B * N

    token_rewards = torch.zeros(total, seq_len)
    response_mask = torch.ones(total, seq_len)

    # Assign varied rewards at last token position
    for i in range(total):
        token_rewards[i, -1] = float(i % 5) / 5.0

    uid = np.repeat(np.arange(B), N)

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        # Fresh DataProto for each call (compute_advantage mutates in place)
        batch = TensorDict({
            "token_level_rewards": token_rewards.clone(),
            "response_mask": response_mask.clone(),
        }, batch_size=[total])

        data = DataProto(
            batch=batch,
            non_tensor_batch={"uid": uid.copy()},
        )

        estimator = AdvantageEstimator(name)
        result = compute_advantage(data, estimator)

        assert "advantages" in result.batch.keys(), \
            f"{name}: advantages missing from result"
        assert "returns" in result.batch.keys(), \
            f"{name}: returns missing from result"
        assert result.batch["advantages"].shape == (total, seq_len), \
            f"{name}: wrong advantages shape"
        assert result.batch["returns"].shape == (total, seq_len), \
            f"{name}: wrong returns shape"
        assert not torch.isnan(result.batch["advantages"]).any(), \
            f"{name}: NaN in advantages"
        assert not torch.isnan(result.batch["returns"]).any(), \
            f"{name}: NaN in returns"


# ── Qwen3 native vLLM support tests ──────────────────────────────────────


class TestQwen3VllmNativeSupport:
    """Verify the backported Qwen3 model is properly registered in vLLM.

    We backported qwen3.py from vLLM 0.8.5 into the existing vLLM 0.8.3
    install to get native (non-Transformers-fallback) Qwen3 inference.
    These tests ensure the backport stays correct across env changes.
    """

    def test_qwen3_resolves_to_native_model(self):
        """Qwen3ForCausalLM must resolve to native vLLM model, not TransformersForCausalLM."""
        from vllm.model_executor.models.registry import ModelRegistry
        cls, name = ModelRegistry.resolve_model_cls(["Qwen3ForCausalLM"])
        assert "TransformersForCausalLM" not in str(cls), \
            f"Qwen3 fell back to TransformersForCausalLM (slow path): {cls}"
        assert "qwen3" in cls.__module__, \
            f"Qwen3 resolved to wrong module: {cls.__module__}"

    def test_qwen3_model_importable(self):
        """qwen3.py module must be importable from vLLM models."""
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        assert Qwen3ForCausalLM is not None

    def test_qwen3_has_qk_norm(self):
        """Qwen3Attention must have q_norm and k_norm (RMSNorm on Q/K).

        This is THE critical architectural difference vs Qwen2.
        Without qk_norm, outputs are numerically wrong.
        """
        from vllm.model_executor.models.qwen3 import Qwen3Attention
        src = inspect.getsource(Qwen3Attention)
        assert "q_norm" in src, "Qwen3Attention missing q_norm"
        assert "k_norm" in src, "Qwen3Attention missing k_norm"

    def test_qwen3_attention_no_bias(self):
        """Qwen3 attention projections must use bias=False (unlike Qwen2)."""
        from vllm.model_executor.models.qwen3 import Qwen3Attention
        src = inspect.getsource(Qwen3Attention.__init__)
        assert "bias=False" in src, \
            "Qwen3Attention should have bias=False for QKV projections"

    def test_qwen3_has_load_weights(self):
        """Qwen3ForCausalLM must have a load_weights method."""
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        assert hasattr(Qwen3ForCausalLM, "load_weights")

    def test_qwen3_reuses_qwen2_mlp(self):
        """Qwen3 MLP should be aliased from Qwen2 (identical architecture)."""
        from vllm.model_executor.models.qwen3 import Qwen3MLP
        from vllm.model_executor.models.qwen2 import Qwen2MLP
        assert Qwen3MLP is Qwen2MLP, \
            "Qwen3MLP should be Qwen2MLP (same architecture)"

    def test_qwen3_registry_entry_correct(self):
        """Registry must map Qwen3ForCausalLM to ('qwen3', 'Qwen3ForCausalLM')."""
        from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
        assert "Qwen3ForCausalLM" in _TEXT_GENERATION_MODELS, \
            "Qwen3ForCausalLM not in _TEXT_GENERATION_MODELS registry"
        mod, arch = _TEXT_GENERATION_MODELS["Qwen3ForCausalLM"]
        assert mod == "qwen3", f"Expected module 'qwen3', got '{mod}'"
        assert arch == "Qwen3ForCausalLM", f"Expected arch 'Qwen3ForCausalLM', got '{arch}'"

    def test_qwen3_all_imports_valid(self):
        """All imports used by qwen3.py must exist in vLLM 0.8.3."""
        # These are the key imports from qwen3.py
        from vllm.attention import Attention, AttentionType
        from vllm.compilation.decorators import support_torch_compile
        from vllm.config import CacheConfig, VllmConfig
        from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.sampling_metadata import SamplingMetadata
        from vllm.sequence import IntermediateTensors
        from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

    def test_qwen3_model_no_decoder_layer_type_kwarg(self):
        """Qwen3Model must NOT pass decoder_layer_type to Qwen2Model.__init__.

        vLLM 0.8.3's Qwen2Model doesn't accept decoder_layer_type.
        The backported qwen3.py must override __init__ to build layers directly.
        """
        from vllm.model_executor.models.qwen2 import Qwen2Model
        import inspect
        qwen2_sig = inspect.signature(Qwen2Model.__init__)
        assert "decoder_layer_type" not in qwen2_sig.parameters, \
            "If Qwen2Model now accepts decoder_layer_type, the backport workaround can be simplified"

    def test_qwen3_has_sample_method(self):
        """Qwen3ForCausalLM must have a sample() method.

        Without this, VllmModelForTextGeneration protocol check fails and
        LLM.generate() raises ValueError (runner_type != 'generate').
        """
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        assert hasattr(Qwen3ForCausalLM, "sample"), \
            "Qwen3ForCausalLM missing sample() — LLM.generate() will fail"
        sig = inspect.signature(Qwen3ForCausalLM.sample)
        params = list(sig.parameters.keys())
        assert "logits" in params, "sample() must accept logits parameter"
        assert "sampling_metadata" in params, "sample() must accept sampling_metadata"

    def test_qwen3_has_sampler_init(self):
        """Qwen3ForCausalLM.__init__ must initialize self.sampler via get_sampler().

        The sample() method calls self.sampler(), so it must be initialized.
        """
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        src = inspect.getsource(Qwen3ForCausalLM.__init__)
        assert "get_sampler()" in src, \
            "Qwen3ForCausalLM.__init__ must call get_sampler() to init self.sampler"

    def test_qwen3_is_text_generation_model(self):
        """Qwen3ForCausalLM must satisfy VllmModelForTextGeneration protocol.

        This is the runtime check that determines runner_type='generate'.
        Requires both compute_logits() and sample() methods.
        """
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.model_executor.models.interfaces_base import is_text_generation_model
        assert is_text_generation_model(Qwen3ForCausalLM), \
            "Qwen3ForCausalLM fails is_text_generation_model check — LLM.generate() will reject it"

    def test_qwen3_has_compute_logits(self):
        """Qwen3ForCausalLM must have compute_logits() for the generation protocol."""
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        assert hasattr(Qwen3ForCausalLM, "compute_logits")

    def test_qwen3_sampler_import_exists(self):
        """get_sampler and SamplerOutput must be importable (used by qwen3.py)."""
        from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
        assert callable(get_sampler)
        assert SamplerOutput is not None

    def test_qwen3_sample_return_type_annotation(self):
        """sample() must return Optional[SamplerOutput] (matching Qwen2's signature)."""
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        qwen3_sig = inspect.signature(Qwen3ForCausalLM.sample)
        qwen2_sig = inspect.signature(Qwen2ForCausalLM.sample)
        assert qwen3_sig.return_annotation == qwen2_sig.return_annotation, \
            f"Qwen3.sample return type {qwen3_sig.return_annotation} != Qwen2 {qwen2_sig.return_annotation}"

    def test_qwen2_still_works(self):
        """Qwen2 native support must still work after adding Qwen3."""
        from vllm.model_executor.models.registry import ModelRegistry
        cls, name = ModelRegistry.resolve_model_cls(["Qwen2ForCausalLM"])
        assert "qwen2" in cls.__module__
        assert "TransformersForCausalLM" not in str(cls)


# ── Eval batch consolidation tests ────────────────────────────────────

class TestEvalBatchConsolidation:
    """Verify DataProto.concat works correctly for eval batch consolidation.

    The insitu eval now accumulates all eval batches into one large batch
    per pass (instead of calling generate_sequences per small batch).
    These tests ensure the concat + index math is correct.
    """

    def test_dataproto_concat_preserves_data(self):
        """DataProto.concat must preserve all tensor data."""
        from verl import DataProto
        from tensordict import TensorDict

        batch1 = DataProto(
            batch=TensorDict({"input_ids": torch.ones(4, 10, dtype=torch.long)}, batch_size=[4]),
            non_tensor_batch={"raw_prompt_ids": np.array([list(range(10))] * 4, dtype=object)},
        )
        batch2 = DataProto(
            batch=TensorDict({"input_ids": torch.ones(6, 10, dtype=torch.long) * 2}, batch_size=[6]),
            non_tensor_batch={"raw_prompt_ids": np.array([list(range(10))] * 6, dtype=object)},
        )

        combined = DataProto.concat([batch1, batch2])
        assert len(combined) == 10
        assert combined.batch["input_ids"].shape == (10, 10)
        # First 4 should be 1s, next 6 should be 2s
        assert (combined.batch["input_ids"][:4] == 1).all()
        assert (combined.batch["input_ids"][4:] == 2).all()

    def test_eval_index_math(self):
        """Verify the problem index calculation after batch consolidation.

        After concat, responses are ordered: [batch0_prob0_comp0..comp15, batch0_prob1_comp0..comp15, ..., batch1_prob0_comp0..comp15, ...]
        The index math must correctly map sequence index -> problem index.
        """
        n_per_pass = 16
        batch_sizes = [3, 2]  # 3 problems in first batch, 2 in second
        total_seqs = sum(bs * n_per_pass for bs in batch_sizes)  # 80

        # Simulate the index calculation from ray_trainer._insitu_eval
        problem_assignments = []
        global_problem_idx = 0
        seq_idx = 0
        for bs in batch_sizes:
            for i in range(bs * n_per_pass):
                problem_idx = global_problem_idx + (i // n_per_pass)
                problem_assignments.append(problem_idx)
                seq_idx += 1
            global_problem_idx += bs

        assert len(problem_assignments) == total_seqs
        # Problem 0 should get 16 completions (indices 0-15)
        assert problem_assignments.count(0) == n_per_pass
        # Problem 2 (first of second batch) should also get 16
        assert problem_assignments.count(3) == n_per_pass
        # Total unique problems
        assert len(set(problem_assignments)) == sum(batch_sizes)
