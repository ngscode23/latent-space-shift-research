from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strict_attractor_existence_verifier import (  # noqa: E402
    collect_diagnostics_not_used_as_proof,
    verify,
)


def load_example(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


def finite_base() -> dict:
    return load_example("finite_exists.json")


def contraction_base() -> dict:
    return load_example("contraction_exists.json")


def test_finite_true_attractor_returns_proved() -> None:
    report = verify(finite_base(), "finite_exhaustive")

    assert report["result"]["status"] == "proved"
    assert report["result"]["candidate_status"] == "proved"
    assert report["result"]["global_existence_status"] == "proved"
    assert report["attractors_found"][0]["certificate_type"] == "finite_exhaustive"


def test_empty_A_returns_candidate_refuted() -> None:
    data = finite_base()
    data["A"] = []

    report = verify(data, "finite_exhaustive")

    assert report["result"]["candidate_status"] == "refuted"
    assert any(row["condition_failed"] == "A_nonempty" for row in report["refutations"])


def test_F_A_not_equal_A_returns_candidate_refuted() -> None:
    report = verify(load_example("finite_no_invariance.json"), "finite_exhaustive")

    assert report["result"]["candidate_status"] == "refuted"
    assert any(row["condition_failed"] == "F_A_equals_A" for row in report["refutations"])


def test_non_autonomous_F_is_rejected() -> None:
    data = finite_base()
    data["F"]["autonomous"] = False

    report = verify(data, "finite_exhaustive")

    assert report["result"]["status"] == "not_established"
    assert report["system"]["autonomous"] is False
    assert any(row["condition_failed"] == "F_autonomous" for row in report["refutations"])


def test_nondeterministic_F_is_rejected() -> None:
    data = finite_base()
    data["F"]["mapping"]["a"] = ["a", "b"]

    report = verify(data, "finite_exhaustive")

    assert report["result"]["status"] == "not_established"
    assert report["system"]["deterministic"] is False
    assert any(row["condition_failed"] == "F_deterministic" for row in report["refutations"])


def test_finite_orbit_outside_A_returns_candidate_refuted() -> None:
    report = verify(load_example("finite_no_convergence.json"), "finite_exhaustive")

    assert report["result"]["candidate_status"] == "refuted"
    assert any(
        row["condition_failed"] == "convergence_for_all_x_in_U"
        for row in report["refutations"]
    )


def test_finite_existence_any_finds_cycles() -> None:
    report = verify(load_example("finite_find_cycles.json"), "finite_exhaustive")

    attractor_sets = {tuple(row["A"]) for row in report["attractors_found"]}
    assert report["result"]["status"] == "proved"
    assert report["result"]["candidate_status"] == "not_applicable"
    assert ("a", "b") in attractor_sets
    assert ("e",) in attractor_sets


def test_candidate_refuted_does_not_automatically_refute_global_existence() -> None:
    report = verify(load_example("finite_no_invariance.json"), "finite_exhaustive")

    assert report["result"]["candidate_status"] == "refuted"
    assert report["result"]["global_existence_status"] == "proved"
    assert report["result"]["status"] == "proved"


def test_contraction_certificate_L_less_than_one_returns_proved() -> None:
    report = verify(contraction_base(), "contraction_certificate")

    assert report["result"]["status"] == "proved"
    assert report["result"]["candidate_status"] == "proved"
    assert report["attractors_found"][0]["certificate_type"] == "contraction"


def test_contraction_L_greater_equal_one_returns_not_established() -> None:
    report = verify(
        load_example("contraction_L_ge_1_not_established.json"),
        "contraction_certificate",
    )

    assert report["result"]["status"] == "not_established"
    assert report["result"]["candidate_status"] == "not_established"


def test_sampled_perturbations_never_produce_proved() -> None:
    report = verify(load_example("contraction_not_established.json"), "contraction_certificate")

    assert report["result"]["status"] == "not_established"
    assert report["attractors_found"] == []
    assert "empirical_fields_present_not_used_as_proof" in report["diagnostics_not_used_as_proof"]


def test_old_empirical_llm_metrics_never_produce_proof() -> None:
    report = verify(load_example("llm_empirical_metrics_not_proof.json"), "finite_exhaustive")

    assert report["result"]["status"] == "not_established"
    assert report["attractors_found"] == []
    assert report["forbidden_reasoning_used"] is False
    assert "empirical_fields_present_not_used_as_proof" in report["diagnostics_not_used_as_proof"]


def test_missing_X_d_F_returns_not_established() -> None:
    report = verify({"query_type": "candidate_attractor"}, "finite_exhaustive")

    assert report["result"]["status"] == "not_established"
    assert report["result"]["candidate_status"] == "not_established"


def test_centroid_only_A_without_invariance_proof_is_not_established() -> None:
    data = contraction_base()
    data["A"] = {
        "type": "centroid",
        "point": [0.0]
    }
    data["certificate"] = copy.deepcopy(data["certificate"])
    data["certificate"].pop("fixed_point", None)

    report = verify(data, "contraction_certificate")

    assert report["result"]["status"] == "not_established"
    assert report["result"]["candidate_status"] == "not_established"


def test_external_free_text_certificate_does_not_produce_proved() -> None:
    data = contraction_base()
    data["certificate"] = {
        "type": "external",
        "text": "A theorem proves that this is a contraction."
    }

    report = verify(data, "contraction_certificate")

    assert report["result"]["status"] == "not_established"
    assert report["system"]["external_certificate_trusted"] is False
    assert "external_or_textual_certificate_not_trusted" in report["diagnostics_not_used_as_proof"]


def test_missing_U_completeness_in_contraction_mode_returns_not_established() -> None:
    data = contraction_base()
    data["certificate"].pop("U_complete")

    report = verify(data, "contraction_certificate")

    assert report["result"]["status"] == "not_established"
    assert report["attractors_found"] == []


def test_missing_genuine_neighborhood_witness_in_euclidean_mode_returns_not_established() -> None:
    data = contraction_base()
    data["certificate"].pop("neighborhood_witness_epsilon")

    report = verify(data, "contraction_certificate")

    assert report["result"]["status"] == "not_established"
    assert report["attractors_found"] == []


def test_diagnostics_functions_cannot_write_proof_fields() -> None:
    data = {
        "query_type": "candidate_attractor",
        "diagnostics": {
            "result": {"status": "proved"},
            "candidate_status": "proved",
            "strict_gates": {"passed": True}
        }
    }

    diagnostics = collect_diagnostics_not_used_as_proof(data, "finite_exhaustive")
    report = verify(data, "finite_exhaustive")

    assert diagnostics["input_diagnostics"]["result"]["status"] == "proved"
    assert report["result"]["status"] == "not_established"
    assert report["result"]["candidate_status"] == "not_established"
    assert report["attractors_found"] == []
    assert report["forbidden_reasoning_used"] is False
