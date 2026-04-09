"""DDP correctness tests.

The rank-based test must be run via:
    torchrun --standalone --nproc_per_node=2 -m pytest experiment7_depth/tests/test_ddp_correctness.py -v

The non-DDP tests run normally with pytest.
"""

import os
import torch
import torch.nn as nn
import pytest
from experiment7_depth.model import create_model, convert_decoder_bns_to_sync


def test_syncbn_conversion_decoder():
    """Verify decoder BNs are converted to SyncBatchNorm."""
    model = create_model(K=100, method="ordinal_ce", pretrained=False, seed=0)
    model = convert_decoder_bns_to_sync(model)

    for name in ["dec1", "dec2", "dec3", "dec4", "dec5"]:
        dec = getattr(model, name)
        for m in dec.modules():
            if "BatchNorm" in type(m).__name__:
                assert "SyncBatchNorm" in type(m).__name__, \
                    f"Decoder {name} BN not converted: {type(m).__name__}"


def test_syncbn_encoder_unchanged():
    """Verify encoder BNs remain plain BatchNorm2d."""
    model = create_model(K=100, method="ordinal_ce", pretrained=False, seed=0)
    model = convert_decoder_bns_to_sync(model)

    for layer in [model.enc_layer2, model.enc_layer3, model.enc_layer4]:
        for m in layer.modules():
            if "BatchNorm" in type(m).__name__:
                assert type(m).__name__ == "BatchNorm2d", \
                    f"Encoder BN should not be SyncBN: {type(m).__name__}"


def test_run_module_imports():
    """Verify run.py imports cleanly."""
    from experiment7_depth.run import main, parse_args, setup_ddp, is_main_process
    assert callable(main)
    assert callable(parse_args)


def test_ddp_2rank_loss_finite():
    """2-rank DDP test. Skip if not launched via torchrun."""
    if "RANK" not in os.environ:
        pytest.skip("Not in DDP context; run via torchrun --nproc_per_node=2")

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from experiment7_depth.losses import dense_ordinal_ce_loss

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    K = 100
    model = create_model(K=K, method="ordinal_ce", pretrained=False, seed=0).to(device)
    model = convert_decoder_bns_to_sync(model)
    model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, momentum=0.9, weight_decay=5e-4
    )

    for step in range(5):
        torch.manual_seed(step * 100 + local_rank)
        images = torch.randn(4, 3, 64, 64, device=device)
        target_bin = torch.randint(0, K, (4, 64, 64), device=device)
        valid_mask = torch.ones(4, 64, 64, dtype=torch.bool, device=device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = dense_ordinal_ce_loss(logits, target_bin, valid_mask, K)
        loss.backward()
        opt.step()

        assert torch.isfinite(loss), f"Step {step}: non-finite loss"

    dist.destroy_process_group()
