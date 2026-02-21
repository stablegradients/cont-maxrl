"""Tests for generate_figures.py: graceful degradation with missing/malformed data.

Covers bugs:
  R1-9: generate_figures.py crashes on empty results
  R4-1: figure_c KeyError on missing per_problem key
"""
import json
import os
from pathlib import Path

import pytest


# ── load_eval_results ──


def test_load_eval_results_empty_dir(tmp_path):
    """load_eval_results returns {} for nonexistent results."""
    from experiment4_llm.scripts.generate_figures import load_eval_results
    result = load_eval_results(str(tmp_path), "7b", "apps", "maxrl_unnorm", 16, "apps")
    assert result == {}


def test_load_eval_results_single_seed(tmp_path):
    """load_eval_results works with a single seed's data."""
    from experiment4_llm.scripts.generate_figures import load_eval_results

    exp_dir = tmp_path / "7b_apps_maxrl_unnorm_g16_s42" / "eval_apps"
    os.makedirs(exp_dir)
    result_data = {
        "pass_at_k": {"1": 0.5, "10": 0.7},
        "per_problem": [{"problem_idx": 0, "n": 200, "c": 100, "pass_at_k": {"1": 0.5}}],
        "summary": {"total_problems": 1},
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(result_data, f)

    data = load_eval_results(str(tmp_path), "7b", "apps", "maxrl_unnorm", 16, "apps")
    assert data != {}
    assert 1 in data["avg"]
    assert data["avg"][1] == pytest.approx(0.5)
    assert data["n_seeds"] == 1


def test_load_eval_results_malformed_no_per_problem(tmp_path):
    """load_eval_results handles JSON without per_problem key (BUG R4-1)."""
    from experiment4_llm.scripts.generate_figures import load_eval_results

    exp_dir = tmp_path / "7b_apps_maxrl_unnorm_g16_s42" / "eval_apps"
    os.makedirs(exp_dir)
    result_data = {
        "pass_at_k": {"1": 0.5, "10": 0.7},
        "summary": {"total_problems": 10},
        # per_problem intentionally missing
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(result_data, f)

    data = load_eval_results(str(tmp_path), "7b", "apps", "maxrl_unnorm", 16, "apps")
    assert data != {}
    assert data["avg"][1] == pytest.approx(0.5)
    # "raw" should exist and contain the result
    assert len(data["raw"]) == 1


# ── Figure functions: no crash on empty data ──


def test_figure_a_no_data_no_crash(tmp_path):
    """figure_a should not crash when results_dir is empty."""
    from experiment4_llm.scripts.generate_figures import figure_a_pass_at_k

    results_dir = str(tmp_path / "empty")
    output_dir = str(tmp_path / "output")
    os.makedirs(results_dir)
    os.makedirs(output_dir)

    # Should silently return, not raise
    figure_a_pass_at_k(results_dir, output_dir, "7b", "apps", 16, "apps")
    # No PDF should be created
    assert len(list(Path(output_dir).glob("*.pdf"))) == 0


def test_figure_c_no_data_no_crash(tmp_path):
    """figure_c should not crash when results_dir is empty."""
    from experiment4_llm.scripts.generate_figures import figure_c_reward_distribution

    results_dir = str(tmp_path / "empty")
    output_dir = str(tmp_path / "output")
    os.makedirs(results_dir)
    os.makedirs(output_dir)

    figure_c_reward_distribution(results_dir, output_dir, "7b", "apps", 16, "apps")
    assert len(list(Path(output_dir).glob("*.pdf"))) == 0


def test_figure_d_no_data_no_crash(tmp_path):
    """figure_d should not crash when results_dir is empty."""
    from experiment4_llm.scripts.generate_figures import figure_d_norm_vs_unnorm

    results_dir = str(tmp_path / "empty")
    output_dir = str(tmp_path / "output")
    os.makedirs(results_dir)
    os.makedirs(output_dir)

    figure_d_norm_vs_unnorm(results_dir, output_dir, "7b", "apps", 16)
    assert len(list(Path(output_dir).glob("*.pdf"))) == 0


def test_figure_e_no_data_no_crash(tmp_path):
    """figure_e should not crash when results_dir is empty."""
    from experiment4_llm.scripts.generate_figures import figure_e_scaling_with_g

    results_dir = str(tmp_path / "empty")
    output_dir = str(tmp_path / "output")
    os.makedirs(results_dir)
    os.makedirs(output_dir)

    figure_e_scaling_with_g(results_dir, output_dir, "7b", "apps", "apps")
    assert len(list(Path(output_dir).glob("*.pdf"))) == 0


def test_figure_c_missing_per_problem_no_crash(tmp_path):
    """figure_c should not crash when per_problem key is absent (BUG R4-1)."""
    from experiment4_llm.scripts.generate_figures import figure_c_reward_distribution

    # Create result file without per_problem
    exp_dir = tmp_path / "7b_apps_maxrl_unnorm_g16_s42" / "eval_apps"
    os.makedirs(exp_dir)
    result_data = {
        "pass_at_k": {"1": 0.5, "10": 0.7},
        "summary": {"total_problems": 10},
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(result_data, f)

    output_dir = str(tmp_path / "figures_out")
    os.makedirs(output_dir)

    # Should not raise KeyError
    figure_c_reward_distribution(str(tmp_path), output_dir, "7b", "apps", 16, "apps")


# ── Figure functions: produce output with valid data ──


def test_figure_a_produces_pdf_with_valid_data(tmp_path):
    """figure_a should produce a PDF when valid eval results exist."""
    from experiment4_llm.scripts.generate_figures import figure_a_pass_at_k

    # Create valid eval results for one method
    exp_dir = tmp_path / "7b_apps_maxrl_unnorm_g16_s42" / "eval_apps"
    os.makedirs(exp_dir)
    result_data = {
        "pass_at_k": {str(k): 0.1 * i for i, k in enumerate([1, 2, 5, 10, 20, 50, 100])},
        "per_problem": [],
        "summary": {"total_problems": 100},
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(result_data, f)

    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir)
    figure_a_pass_at_k(str(tmp_path), output_dir, "7b", "apps", 16, "apps")

    pdfs = list(Path(output_dir).glob("*.pdf"))
    assert len(pdfs) == 1
    assert "fig_a" in pdfs[0].name


def test_figure_c_produces_pdf_with_valid_data(tmp_path):
    """figure_c should produce a PDF when valid per_problem data exists."""
    from experiment4_llm.scripts.generate_figures import figure_c_reward_distribution

    exp_dir = tmp_path / "7b_apps_maxrl_unnorm_g16_s42" / "eval_apps"
    os.makedirs(exp_dir)
    result_data = {
        "pass_at_k": {"1": 0.5, "10": 0.7},
        "per_problem": [
            {"problem_idx": i, "n": 200, "c": i * 20, "pass_at_k": {"1": i * 0.1}}
            for i in range(10)
        ],
        "summary": {"total_problems": 10},
    }
    with open(exp_dir / "eval_results.json", "w") as f:
        json.dump(result_data, f)

    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir)
    figure_c_reward_distribution(str(tmp_path), output_dir, "7b", "apps", 16, "apps")

    pdfs = list(Path(output_dir).glob("*.pdf"))
    assert len(pdfs) == 1
    assert "fig_c" in pdfs[0].name
