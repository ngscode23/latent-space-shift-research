from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SUPPORTED_MODES = {
    "finite_exhaustive",
    "contraction_certificate",
    "lyapunov_certificate",
    "trapping_region_certificate",
}

PROOF_VALUES = {"proved", "refuted", "missing"}
FORBIDDEN_EMPIRICAL_KEYS = {
    "centroid",
    "centroids",
    "target_centroid",
    "control_centroid",
    "centroid_distance",
    "target_control_separation",
    "hidden_state_clustering",
    "hidden_state_distances",
    "hidden_distances",
    "recovery_score",
    "bootstrap_ci",
    "bootstrap_confidence_intervals",
    "random_perturbation_sampling",
    "sampled_perturbations",
    "basin_sweep",
    "finite_n_convergence",
    "empirical_contraction",
    "negative_slope",
    "local_jacobian_estimates",
    "sampled_jacobian_directions",
    "strict_gates",
    "llm_text_behavior",
    "target_txt",
    "target_txt_reference",
    "target_texts",
}


CONDITION_KEYS = [
    "A_nonempty",
    "A_subset_X",
    "A_compact",
    "F_well_defined",
    "F_autonomous",
    "F_deterministic",
    "F_A_equals_A",
    "U_contains_A",
    "U_subset_X",
    "U_is_neighborhood",
    "convergence_for_all_x_in_U",
]


def condition_template() -> dict[str, str]:
    return {key: "missing" for key in CONDITION_KEYS}


def base_report(query_type: str = "existence_any") -> dict[str, Any]:
    if query_type not in {"candidate_attractor", "existence_any"}:
        query_type = "existence_any"
    return {
        "theorem": "strict mathematical attractor existence",
        "definition": (
            "A nonempty compact invariant set A with an attracting neighborhood U "
            "such that every x in U converges to A"
        ),
        "query_type": query_type,
        "system": {
            "state_space_X": "missing",
            "metric_d": "missing",
            "transition_F": "missing",
            "autonomous": False,
            "deterministic": False,
            "external_certificate_trusted": False,
        },
        "result": {
            "status": "not_established",
            "candidate_status": "not_applicable"
            if query_type == "existence_any"
            else "not_established",
            "global_existence_status": "not_established",
            "meaning": "No strict mathematical attractor proof has been established.",
        },
        "attractors_found": [],
        "refutations": [],
        "diagnostics_not_used_as_proof": {},
        "forbidden_reasoning_used": False,
    }


def add_refutation(
    report: dict[str, Any],
    condition_failed: str,
    counterexample: Any = "",
    details: str = "",
) -> None:
    report["refutations"].append(
        {
            "condition_failed": condition_failed,
            "counterexample": counterexample,
            "details": details,
        }
    )


def decimal_from_json(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def decimal_positive(value: Any) -> bool:
    parsed = decimal_from_json(value)
    return parsed is not None and parsed > 0


def decimal_lt_one(value: Any) -> bool:
    parsed = decimal_from_json(value)
    return parsed is not None and parsed < Decimal("1")


def decimal_ge_one(value: Any) -> bool:
    parsed = decimal_from_json(value)
    return parsed is not None and parsed >= Decimal("1")


def decimal_eq(a: Any, b: Any) -> bool:
    da = decimal_from_json(a)
    db = decimal_from_json(b)
    return da is not None and db is not None and da == db


def is_json_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def canonical_state(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def stringify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def collect_empirical_keys(value: Any, path: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_EMPIRICAL_KEYS:
                found[child_path] = child
            found.update(collect_empirical_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(collect_empirical_keys(child, f"{path}[{index}]"))
    return found


def collect_diagnostics_not_used_as_proof(data: Any, mode: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"mode_requested": mode}
    if not isinstance(data, dict):
        diagnostics["input_type"] = type(data).__name__
        return diagnostics

    schema_keys = {
        "query_type",
        "system_type",
        "X",
        "dimension",
        "metric",
        "F",
        "A",
        "U",
        "certificate",
        "diagnostics",
    }
    unknown = {key: value for key, value in data.items() if key not in schema_keys}
    if unknown:
        diagnostics["unknown_fields_preserved"] = unknown
    if isinstance(data.get("diagnostics"), dict):
        diagnostics["input_diagnostics"] = data["diagnostics"]

    empirical = collect_empirical_keys(data)
    if empirical:
        diagnostics["empirical_fields_present_not_used_as_proof"] = empirical

    certificate = data.get("certificate")
    if isinstance(certificate, dict):
        cert_type = certificate.get("type")
        if cert_type in {"external", "free_text", "textual"} or "text" in certificate:
            diagnostics["external_or_textual_certificate_not_trusted"] = certificate
    if "external_certificate" in data:
        diagnostics["external_certificate_not_trusted"] = data["external_certificate"]
    if "free_text_certificate" in data:
        diagnostics["free_text_certificate_not_trusted"] = data["free_text_certificate"]
    return diagnostics


@dataclass
class StateSpace:
    system_type: str
    states: list[str] | None = None
    dimension: int | None = None

    @classmethod
    def from_data(cls, data: dict[str, Any], report: dict[str, Any]) -> "StateSpace | None":
        system_type = data.get("system_type")
        if system_type == "finite":
            raw_states = data.get("X")
            if not isinstance(raw_states, list):
                add_refutation(report, "X", "", "Finite mode requires X as an explicit finite list.")
                return None
            states = [canonical_state(item) for item in raw_states]
            if len(states) != len(set(states)):
                add_refutation(report, "X", "", "Finite state list contains duplicates.")
                return None
            report["system"]["state_space_X"] = f"finite explicit list with {len(states)} states"
            return cls(system_type="finite", states=states)

        if system_type == "euclidean_certified":
            dimension = data.get("dimension")
            if not isinstance(dimension, int) or dimension <= 0:
                add_refutation(
                    report,
                    "X",
                    "",
                    "Euclidean certified mode requires a positive integer dimension.",
                )
                return None
            report["system"]["state_space_X"] = f"euclidean_certified subset of R^{dimension}"
            return cls(system_type="euclidean_certified", dimension=dimension)

        add_refutation(report, "system_type", system_type, "Unsupported or missing formal system_type.")
        return None

    def contains_finite_set(self, values: set[str]) -> bool:
        return self.states is not None and values.issubset(set(self.states))

    def valid_point(self, point: Any) -> bool:
        if self.dimension is None or not isinstance(point, list) or len(point) != self.dimension:
            return False
        return all(decimal_from_json(value) is not None for value in point)


@dataclass
class Metric:
    metric_type: str
    values: dict[str, Any] | None = None

    @classmethod
    def from_data(cls, data: dict[str, Any], space: StateSpace, report: dict[str, Any]) -> "Metric | None":
        raw = data.get("metric")
        if not isinstance(raw, dict):
            add_refutation(report, "metric_d", "", "Metric d is missing.")
            return None
        metric_type = raw.get("type")
        if space.system_type == "finite":
            if metric_type == "discrete":
                report["system"]["metric_d"] = "discrete metric"
                return cls(metric_type="discrete")
            if metric_type == "table":
                if not isinstance(raw.get("values"), dict):
                    add_refutation(report, "metric_d", "", "Metric table requires a values object.")
                    return None
                report["system"]["metric_d"] = "finite metric table"
                return cls(metric_type="table", values=raw["values"])
        if space.system_type == "euclidean_certified" and metric_type == "l2":
            report["system"]["metric_d"] = "l2 metric"
            return cls(metric_type="l2")
        add_refutation(report, "metric_d", metric_type, "Unsupported metric for this state space.")
        return None

    def distance_to_set(self, x: str, A: set[str]) -> int:
        if self.metric_type == "discrete":
            return 0 if x in A else 1
        return 0 if x in A else 1


@dataclass
class FormalSystem:
    space: StateSpace
    metric: Metric
    F_raw: dict[str, Any]
    mapping: dict[str, str] | None = None

    @classmethod
    def from_data(
        cls,
        data: dict[str, Any],
        space: StateSpace,
        metric: Metric,
        report: dict[str, Any],
    ) -> "FormalSystem | None":
        F_raw = data.get("F")
        if not isinstance(F_raw, dict):
            add_refutation(report, "transition_F", "", "Transition F is missing.")
            return None

        report["system"]["transition_F"] = str(F_raw.get("type") or F_raw.get("representation") or "specified")

        if is_non_autonomous(F_raw, data):
            report["system"]["autonomous"] = False
            add_refutation(report, "F_autonomous", F_raw, "F is not one autonomous transition operator.")
            return None
        report["system"]["autonomous"] = True

        if is_nondeterministic(F_raw):
            report["system"]["deterministic"] = False
            add_refutation(report, "F_deterministic", F_raw, "F is stochastic or nondeterministic.")
            return None
        report["system"]["deterministic"] = True

        if space.system_type == "finite":
            if F_raw.get("type") != "total_mapping" or not isinstance(F_raw.get("mapping"), dict):
                add_refutation(report, "F_well_defined", F_raw, "Finite mode requires F.type=total_mapping.")
                return None
            mapping = {
                canonical_state(key): canonical_state(value)
                for key, value in F_raw["mapping"].items()
                if is_json_scalar(value)
            }
            states = set(space.states or [])
            if set(mapping.keys()) != states:
                add_refutation(
                    report,
                    "F_well_defined",
                    sorted(states - set(mapping.keys())),
                    "F mapping must be total on X.",
                )
                return None
            bad_images = {key: value for key, value in mapping.items() if value not in states}
            if bad_images:
                add_refutation(report, "F_well_defined", bad_images, "F(x) must lie in X for every x.")
                return None
            return cls(space=space, metric=metric, F_raw=F_raw, mapping=mapping)

        representation = F_raw.get("representation")
        if representation not in {"affine", "verified"}:
            add_refutation(
                report,
                "F_well_defined",
                F_raw,
                "Euclidean certified mode requires a recognized deterministic representation.",
            )
            return None
        return cls(space=space, metric=metric, F_raw=F_raw)


def is_non_autonomous(F_raw: dict[str, Any], data: dict[str, Any]) -> bool:
    if F_raw.get("autonomous") is False:
        return True
    if F_raw.get("changes_across_steps") is True:
        return True
    if F_raw.get("type") in {"sequence", "time_varying", "multiple_operators"}:
        return True
    if any(key in F_raw for key in ["operators", "recovery_prompts", "prompt_templates", "interventions"]):
        return True
    if any(key in data for key in ["recovery_prompts", "prompt_templates", "intervention_procedures"]):
        return True
    return False


def is_nondeterministic(F_raw: dict[str, Any]) -> bool:
    if F_raw.get("deterministic") is False:
        return True
    if F_raw.get("stochastic") is True or F_raw.get("random") is True:
        return True
    decoding = F_raw.get("decoding")
    if isinstance(decoding, dict):
        if decimal_from_json(decoding.get("temperature", 0)) not in {None, Decimal("0")}:
            return True
        if any(key in decoding for key in ["top_p", "top_k", "random_seed"]):
            return True
    if "temperature" in F_raw and decimal_from_json(F_raw.get("temperature")) != Decimal("0"):
        return True
    if any(key in F_raw for key in ["top_p", "top_k", "random_decoding"]):
        return True
    mapping = F_raw.get("mapping")
    if isinstance(mapping, dict) and any(isinstance(value, list) for value in mapping.values()):
        return True
    return False


@dataclass
class AttractorCandidate:
    raw: Any
    finite_set: set[str] | None = None
    point: list[Any] | None = None
    is_centroid: bool = False

    @classmethod
    def from_data(
        cls,
        data: dict[str, Any],
        space: StateSpace,
        report: dict[str, Any],
        conditions: dict[str, str],
    ) -> "AttractorCandidate | None":
        if "A" not in data:
            return None
        raw = data.get("A")
        if space.system_type == "finite":
            if not isinstance(raw, list):
                conditions["A_nonempty"] = "refuted"
                add_refutation(report, "A_nonempty", raw, "Finite candidate A must be a list of states.")
                return None
            finite_set = {canonical_state(item) for item in raw}
            if not finite_set:
                conditions["A_nonempty"] = "refuted"
                add_refutation(report, "A_nonempty", raw, "Candidate A is empty.")
                return cls(raw=raw, finite_set=finite_set)
            conditions["A_nonempty"] = "proved"
            if space.contains_finite_set(finite_set):
                conditions["A_subset_X"] = "proved"
                conditions["A_compact"] = "proved"
            else:
                conditions["A_subset_X"] = "refuted"
                add_refutation(report, "A_subset_X", sorted(finite_set), "Candidate A is not a subset of X.")
            return cls(raw=raw, finite_set=finite_set)

        if isinstance(raw, dict) and raw.get("type") in {"singleton", "centroid"}:
            point = raw.get("point")
            is_centroid = raw.get("type") == "centroid"
            if not space.valid_point(point):
                conditions["A_nonempty"] = "refuted"
                add_refutation(report, "A_nonempty", raw, "Euclidean singleton candidate must contain a valid point.")
                return None
            conditions["A_nonempty"] = "proved"
            if is_centroid and raw.get("centroid_is_state") is not True:
                conditions["A_subset_X"] = "missing"
            else:
                conditions["A_subset_X"] = "proved"
            conditions["A_compact"] = "proved"
            return cls(raw=raw, point=point, is_centroid=is_centroid)

        conditions["A_nonempty"] = "missing"
        add_refutation(report, "A", raw, "Unsupported candidate A schema.")
        return None

    def repr(self) -> Any:
        if self.finite_set is not None:
            return sorted(self.finite_set)
        return self.raw


@dataclass
class Neighborhood:
    raw: Any
    finite_set: set[str] | None = None

    @classmethod
    def from_data(
        cls,
        data: dict[str, Any],
        space: StateSpace,
        candidate: AttractorCandidate | None,
        report: dict[str, Any],
        conditions: dict[str, str],
        certificate: dict[str, Any] | None = None,
    ) -> "Neighborhood | None":
        raw = data.get("U")
        if raw is None:
            return None
        if space.system_type == "finite":
            if not isinstance(raw, list):
                conditions["U_subset_X"] = "refuted"
                add_refutation(report, "U_subset_X", raw, "Finite U must be a list of states.")
                return None
            finite_set = {canonical_state(item) for item in raw}
            if space.contains_finite_set(finite_set):
                conditions["U_subset_X"] = "proved"
            else:
                conditions["U_subset_X"] = "refuted"
                add_refutation(report, "U_subset_X", sorted(finite_set), "U is not a subset of X.")
            if candidate and candidate.finite_set is not None and candidate.finite_set.issubset(finite_set):
                conditions["U_contains_A"] = "proved"
                conditions["U_is_neighborhood"] = "proved"
            elif candidate:
                conditions["U_contains_A"] = "refuted"
                add_refutation(report, "U_contains_A", raw, "U does not contain A.")
            return cls(raw=raw, finite_set=finite_set)

        if not isinstance(raw, dict) or raw.get("type") not in {"closed_ball", "closed_box"}:
            conditions["U_subset_X"] = "missing"
            add_refutation(report, "U", raw, "Euclidean U must be a closed ball or closed box.")
            return None

        conditions["U_subset_X"] = "proved"
        if candidate and candidate.point is not None:
            if point_in_region(candidate.point, raw):
                conditions["U_contains_A"] = "proved"
            else:
                conditions["U_contains_A"] = "refuted"
                add_refutation(report, "U_contains_A", raw, "Candidate point is not in U.")
        if certificate and candidate and candidate.point is not None:
            epsilon = certificate.get("neighborhood_witness_epsilon")
            if decimal_positive(epsilon) and epsilon_witness_inside_region(candidate.point, raw, epsilon):
                conditions["U_is_neighborhood"] = "proved"
            else:
                conditions["U_is_neighborhood"] = "missing"
        return cls(raw=raw)

    def repr(self) -> Any:
        if self.finite_set is not None:
            return sorted(self.finite_set)
        return self.raw


def point_in_region(point: list[Any], region: dict[str, Any]) -> bool:
    if region.get("type") == "closed_ball":
        center = region.get("center")
        radius = decimal_from_json(region.get("radius"))
        if not isinstance(center, list) or len(center) != len(point) or radius is None or radius < 0:
            return False
        if all(decimal_eq(a, b) for a, b in zip(point, center)):
            return True
        squared = Decimal("0")
        for value, c_value in zip(point, center):
            dv = decimal_from_json(value)
            dc = decimal_from_json(c_value)
            if dv is None or dc is None:
                return False
            squared += (dv - dc) * (dv - dc)
        return squared <= radius * radius

    if region.get("type") == "closed_box":
        lower = region.get("lower")
        upper = region.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list) or len(lower) != len(point):
            return False
        for value, lo, hi in zip(point, lower, upper):
            dv = decimal_from_json(value)
            dlo = decimal_from_json(lo)
            dhi = decimal_from_json(hi)
            if dv is None or dlo is None or dhi is None or not (dlo <= dv <= dhi):
                return False
        return True
    return False


def epsilon_witness_inside_region(point: list[Any], region: dict[str, Any], epsilon: Any) -> bool:
    eps = decimal_from_json(epsilon)
    if eps is None or eps <= 0:
        return False
    if region.get("type") == "closed_ball":
        center = region.get("center")
        radius = decimal_from_json(region.get("radius"))
        if not isinstance(center, list) or radius is None:
            return False
        if all(decimal_eq(a, b) for a, b in zip(point, center)):
            return eps <= radius
        return False
    if region.get("type") == "closed_box":
        lower = region.get("lower")
        upper = region.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            return False
        for value, lo, hi in zip(point, lower, upper):
            dv = decimal_from_json(value)
            dlo = decimal_from_json(lo)
            dhi = decimal_from_json(hi)
            if dv is None or dlo is None or dhi is None:
                return False
            if dv - eps < dlo or dv + eps > dhi:
                return False
        return True
    return False


def find_cycles(mapping: dict[str, str], states: list[str]) -> list[list[str]]:
    cycles: dict[tuple[str, ...], list[str]] = {}
    for start in states:
        path: list[str] = []
        index_by_state: dict[str, int] = {}
        current = start
        while current not in index_by_state:
            index_by_state[current] = len(path)
            path.append(current)
            current = mapping[current]
        cycle = path[index_by_state[current] :]
        key = tuple(sorted(cycle))
        cycles[key] = sorted(cycle)
    return [cycles[key] for key in sorted(cycles)]


def orbit_enters_A(mapping: dict[str, str], start: str, A: set[str]) -> tuple[bool, list[str], list[str]]:
    path: list[str] = []
    seen: dict[str, int] = {}
    current = start
    while current not in seen:
        if current in A:
            path.append(current)
            return True, path, []
        seen[current] = len(path)
        path.append(current)
        current = mapping[current]
    return False, path, path[seen[current] :]


def basin_for_set(mapping: dict[str, str], states: list[str], A: set[str]) -> set[str]:
    basin = set()
    for state in states:
        enters, _, _ = orbit_enters_A(mapping, state, A)
        if enters:
            basin.add(state)
    return basin


def finite_attractor_row(A: set[str], U: set[str], certificate: dict[str, Any]) -> dict[str, Any]:
    conditions = {key: "proved" for key in CONDITION_KEYS}
    return {
        "A": sorted(A),
        "U": sorted(U),
        "certificate_type": "finite_exhaustive",
        "conditions": conditions,
        "certificate": certificate,
    }


def verify_finite_exhaustive(data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    space = StateSpace.from_data(data, report)
    if space is None:
        finalize_report(report)
        return report
    metric = Metric.from_data(data, space, report)
    if metric is None:
        finalize_report(report)
        return report
    system = FormalSystem.from_data(data, space, metric, report)
    if system is None or system.mapping is None:
        finalize_report(report)
        return report

    states = space.states or []
    mapping = system.mapping
    cycles = find_cycles(mapping, states) if states else []
    cycle_rows = [
        finite_attractor_row(
            set(cycle),
            basin_for_set(mapping, states, set(cycle)),
            {"cycle": cycle, "exhaustive": True},
        )
        for cycle in cycles
    ]

    query_type = report["query_type"]
    candidate_conditions = condition_template()
    candidate_conditions["F_well_defined"] = "proved"
    candidate_conditions["F_autonomous"] = "proved"
    candidate_conditions["F_deterministic"] = "proved"
    candidate = AttractorCandidate.from_data(data, space, report, candidate_conditions)

    if candidate is None and "A" not in data:
        report["result"]["candidate_status"] = "not_applicable"
        if cycle_rows:
            report["attractors_found"].extend(cycle_rows)
            report["result"]["global_existence_status"] = "proved"
        elif not states:
            report["result"]["global_existence_status"] = "refuted"
            add_refutation(report, "global_existence", [], "Empty finite X has no nonempty attractor.")
        finalize_report(report)
        return report

    if candidate is None or candidate.finite_set is None:
        report["result"]["candidate_status"] = "not_established"
        report["attractors_found"].extend(cycle_rows)
        if cycle_rows:
            report["result"]["global_existence_status"] = "proved"
        finalize_report(report)
        return report

    A = candidate.finite_set
    if not A:
        report["result"]["candidate_status"] = "refuted"
        report["attractors_found"].extend(cycle_rows)
        if cycle_rows:
            report["result"]["global_existence_status"] = "proved"
        finalize_report(report)
        return report

    if candidate_conditions["A_subset_X"] != "proved":
        report["result"]["candidate_status"] = "refuted"
        report["attractors_found"].extend(cycle_rows)
        if cycle_rows:
            report["result"]["global_existence_status"] = "proved"
        finalize_report(report)
        return report

    image_A = {mapping[state] for state in A}
    if image_A == A:
        candidate_conditions["F_A_equals_A"] = "proved"
    else:
        candidate_conditions["F_A_equals_A"] = "refuted"
        add_refutation(
            report,
            "F_A_equals_A",
            {"A": sorted(A), "F_A": sorted(image_A)},
            "Candidate A is not invariant because F(A) != A.",
        )
        report["result"]["candidate_status"] = "refuted"
        report["attractors_found"].extend(cycle_rows)
        if cycle_rows:
            report["result"]["global_existence_status"] = "proved"
        finalize_report(report)
        return report

    U_obj = Neighborhood.from_data(data, space, candidate, report, candidate_conditions)
    if U_obj is None:
        U = basin_for_set(mapping, states, A)
        candidate_conditions["U_contains_A"] = "proved"
        candidate_conditions["U_subset_X"] = "proved"
        candidate_conditions["U_is_neighborhood"] = "proved"
    else:
        U = U_obj.finite_set or set()

    convergence_ok = True
    for state in sorted(U):
        enters, path, cycle = orbit_enters_A(mapping, state, A)
        if not enters:
            convergence_ok = False
            candidate_conditions["convergence_for_all_x_in_U"] = "refuted"
            add_refutation(
                report,
                "convergence_for_all_x_in_U",
                {"x": state, "path": path, "cycle_outside_A": cycle},
                "An orbit from U enters a cycle outside A.",
            )
            break
    if convergence_ok:
        candidate_conditions["convergence_for_all_x_in_U"] = "proved"

    if all(value == "proved" for value in candidate_conditions.values()):
        report["result"]["candidate_status"] = "proved"
        report["result"]["global_existence_status"] = "proved"
        report["attractors_found"].append(
            {
                "A": sorted(A),
                "U": sorted(U),
                "certificate_type": "finite_exhaustive",
                "conditions": candidate_conditions,
                "certificate": {
                    "exhaustive": True,
                    "basin": sorted(U),
                    "orbits_checked": sorted(U),
                    "finite_topology_neighborhood": True,
                },
            }
        )
    else:
        report["result"]["candidate_status"] = (
            "refuted" if "refuted" in candidate_conditions.values() else "not_established"
        )
        report["attractors_found"].extend(cycle_rows)
        if cycle_rows:
            report["result"]["global_existence_status"] = "proved"

    if query_type == "existence_any" and cycle_rows and not report["attractors_found"]:
        report["attractors_found"].extend(cycle_rows)
        report["result"]["global_existence_status"] = "proved"
    finalize_report(report)
    return report


def residual_interval_status(raw_interval: Any) -> str:
    if not isinstance(raw_interval, list) or len(raw_interval) != 2:
        return "missing"
    lo = decimal_from_json(raw_interval[0])
    hi = decimal_from_json(raw_interval[1])
    if lo is None or hi is None or lo > hi:
        return "missing"
    if lo == 0 and hi == 0:
        return "proved"
    if hi < 0 or lo > 0:
        return "refuted"
    return "missing"


def affine_fixed_point_exact(F_raw: dict[str, Any], point: list[Any]) -> bool:
    if F_raw.get("representation") != "affine":
        return False
    data = F_raw.get("data")
    if not isinstance(data, dict):
        return False
    matrix = data.get("matrix")
    bias = data.get("bias")
    if not isinstance(matrix, list) or not isinstance(bias, list):
        return False
    dim = len(point)
    if len(matrix) != dim or len(bias) != dim:
        return False
    point_dec = [decimal_from_json(value) for value in point]
    if any(value is None for value in point_dec):
        return False
    for row, bias_value, point_value in zip(matrix, bias, point_dec):
        if not isinstance(row, list) or len(row) != dim:
            return False
        total = decimal_from_json(bias_value)
        if total is None:
            return False
        for coefficient, coordinate in zip(row, point_dec):
            coeff = decimal_from_json(coefficient)
            if coeff is None or coordinate is None:
                return False
            total += coeff * coordinate
        if point_value is None or total != point_value:
            return False
    return True


def prepare_euclidean_candidate(
    data: dict[str, Any],
    report: dict[str, Any],
    certificate: dict[str, Any] | None,
) -> tuple[StateSpace | None, Metric | None, FormalSystem | None, AttractorCandidate | None, Neighborhood | None, dict[str, str]]:
    conditions = condition_template()
    space = StateSpace.from_data(data, report)
    if space is None:
        return None, None, None, None, None, conditions
    metric = Metric.from_data(data, space, report)
    if metric is None:
        return space, None, None, None, None, conditions
    system = FormalSystem.from_data(data, space, metric, report)
    if system is None:
        return space, metric, None, None, None, conditions
    conditions["F_well_defined"] = "proved"
    conditions["F_autonomous"] = "proved"
    conditions["F_deterministic"] = "proved"
    candidate = AttractorCandidate.from_data(data, space, report, conditions)
    neighborhood = Neighborhood.from_data(data, space, candidate, report, conditions, certificate)
    return space, metric, system, candidate, neighborhood, conditions


def fixed_point_condition(
    system: FormalSystem,
    candidate: AttractorCandidate | None,
    certificate: dict[str, Any],
    conditions: dict[str, str],
    report: dict[str, Any],
) -> None:
    if candidate is None or candidate.point is None:
        return
    fixed_point = certificate.get("fixed_point")
    residual_status = "missing"
    if isinstance(fixed_point, dict) and fixed_point.get("verified") is True:
        residual_status = residual_interval_status(fixed_point.get("residual_interval"))
    if residual_status == "proved" or affine_fixed_point_exact(system.F_raw, candidate.point):
        conditions["F_A_equals_A"] = "proved"
    elif residual_status == "refuted":
        conditions["F_A_equals_A"] = "refuted"
        add_refutation(
            report,
            "F_A_equals_A",
            fixed_point,
            "Certified residual interval excludes zero; candidate is not a fixed point.",
        )


def verify_contraction_certificate(data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    certificate = data.get("certificate") if isinstance(data.get("certificate"), dict) else {}
    if certificate.get("type") != "contraction":
        finalize_report(report)
        return report
    space, metric, system, candidate, neighborhood, conditions = prepare_euclidean_candidate(
        data, report, certificate
    )
    if not all([space, metric, system, candidate, neighborhood]):
        finalize_candidate_from_conditions(report, conditions)
        return report

    fixed_point_condition(system, candidate, certificate, conditions, report)

    F_maps = certificate.get("F_maps_U_into_U")
    F_maps_verified = isinstance(F_maps, dict) and F_maps.get("verified") is True
    U_complete = certificate.get("U_complete") is True
    lipschitz = certificate.get("lipschitz_bound")
    L = lipschitz.get("L") if isinstance(lipschitz, dict) else None
    lipschitz_verified = (
        isinstance(lipschitz, dict)
        and lipschitz.get("universal_over_U") is True
        and lipschitz.get("bound_kind") in {"analytic", "interval_arithmetic", "machine_checkable"}
        and decimal_lt_one(L)
    )

    if not U_complete:
        conditions["U_is_neighborhood"] = "missing"
    if decimal_ge_one(L):
        report["result"]["meaning"] = "A certified contraction proof was not established because L >= 1."

    if (
        F_maps_verified
        and U_complete
        and lipschitz_verified
        and all(
            conditions[key] == "proved"
            for key in [
                "A_nonempty",
                "A_subset_X",
                "A_compact",
                "F_well_defined",
                "F_autonomous",
                "F_deterministic",
                "F_A_equals_A",
                "U_contains_A",
                "U_subset_X",
                "U_is_neighborhood",
            ]
        )
    ):
        conditions["convergence_for_all_x_in_U"] = "proved"
        report["result"]["candidate_status"] = "proved"
        report["result"]["global_existence_status"] = "proved"
        report["attractors_found"].append(
            {
                "A": candidate.repr(),
                "U": neighborhood.repr(),
                "certificate_type": "contraction",
                "conditions": conditions,
                "certificate": {
                    "theorem": "Banach fixed-point theorem",
                    "L": L,
                    "F_maps_U_into_U": F_maps,
                    "U_complete": U_complete,
                    "neighborhood_witness_epsilon": certificate.get("neighborhood_witness_epsilon"),
                },
            }
        )
    else:
        finalize_candidate_from_conditions(report, conditions)

    finalize_report(report)
    return report


def verify_lyapunov_certificate(data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    certificate = data.get("certificate") if isinstance(data.get("certificate"), dict) else {}
    if certificate.get("type") != "lyapunov":
        finalize_report(report)
        return report
    space, metric, system, candidate, neighborhood, conditions = prepare_euclidean_candidate(
        data, report, certificate
    )
    if not all([space, metric, system, candidate, neighborhood]):
        finalize_candidate_from_conditions(report, conditions)
        return report

    if isinstance(certificate.get("A_invariant"), dict) and certificate["A_invariant"].get("verified") is True:
        conditions["F_A_equals_A"] = "proved"
    else:
        fixed_point_condition(system, candidate, certificate, conditions, report)

    F_maps_verified = (
        isinstance(certificate.get("F_maps_U_into_U"), dict)
        and certificate["F_maps_U_into_U"].get("verified") is True
    )
    V_continuous = (
        isinstance(certificate.get("V_continuous_on_U"), dict)
        and certificate["V_continuous_on_U"].get("verified") is True
    )
    positive_definite = (
        isinstance(certificate.get("V_positive_definite_wrt_A"), dict)
        and certificate["V_positive_definite_wrt_A"].get("verified") is True
    )
    distance_control = (
        isinstance(certificate.get("distance_control"), dict)
        and certificate["distance_control"].get("verified") is True
    )
    decrease = certificate.get("decrease")
    geometric_decrease = (
        isinstance(decrease, dict)
        and decrease.get("kind") == "geometric"
        and decrease.get("universal_over_U") is True
        and decimal_positive(decrease.get("eta"))
    )
    lasalle_decrease = (
        isinstance(decrease, dict)
        and decrease.get("kind") == "lasalle"
        and decrease.get("universal_over_U") is True
        and decrease.get("largest_invariant_set_equals_A") is True
    )

    proof_ready = (
        F_maps_verified
        and V_continuous
        and positive_definite
        and distance_control
        and (geometric_decrease or lasalle_decrease)
        and all(
            conditions[key] == "proved"
            for key in [
                "A_nonempty",
                "A_subset_X",
                "A_compact",
                "F_well_defined",
                "F_autonomous",
                "F_deterministic",
                "F_A_equals_A",
                "U_contains_A",
                "U_subset_X",
                "U_is_neighborhood",
            ]
        )
    )
    if proof_ready:
        conditions["convergence_for_all_x_in_U"] = "proved"
        report["result"]["candidate_status"] = "proved"
        report["result"]["global_existence_status"] = "proved"
        report["attractors_found"].append(
            {
                "A": candidate.repr(),
                "U": neighborhood.repr(),
                "certificate_type": "lyapunov",
                "conditions": conditions,
                "certificate": {
                    "theorem": "Lyapunov attractivity theorem",
                    "decrease": decrease,
                    "distance_control": certificate.get("distance_control"),
                    "neighborhood_witness_epsilon": certificate.get("neighborhood_witness_epsilon"),
                },
            }
        )
    else:
        finalize_candidate_from_conditions(report, conditions)
    finalize_report(report)
    return report


def verify_trapping_region_certificate(data: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    certificate = data.get("certificate") if isinstance(data.get("certificate"), dict) else {}
    if certificate.get("type") != "trapping_region":
        finalize_report(report)
        return report
    conditions = condition_template()
    space = StateSpace.from_data(data, report)
    if space is None:
        finalize_report(report)
        return report
    metric = Metric.from_data(data, space, report)
    if metric is None:
        finalize_report(report)
        return report
    system = FormalSystem.from_data(data, space, metric, report)
    if system is None:
        finalize_report(report)
        return report
    conditions["F_well_defined"] = "proved"
    conditions["F_autonomous"] = "proved"
    conditions["F_deterministic"] = "proved"

    U_raw = data.get("U")
    U_valid = isinstance(U_raw, dict) and U_raw.get("type") in {"closed_ball", "closed_box"}
    if U_valid:
        conditions["U_subset_X"] = "proved"
        conditions["U_is_neighborhood"] = "proved"
    else:
        add_refutation(report, "U", U_raw, "Trapping certificate requires a certified region U.")

    compact = (
        isinstance(certificate.get("closure_U_compact"), dict)
        and certificate["closure_U_compact"].get("verified") is True
    )
    continuous = (
        isinstance(certificate.get("F_continuous_on_closure_U"), dict)
        and certificate["F_continuous_on_closure_U"].get("verified") is True
    )
    nonempty = (
        isinstance(certificate.get("U_nonempty"), dict)
        and certificate["U_nonempty"].get("verified") is True
    )
    strict_trapping = (
        isinstance(certificate.get("F_closure_U_subset_interior_U"), dict)
        and certificate["F_closure_U_subset_interior_U"].get("verified") is True
    )

    if U_valid and compact and continuous and nonempty and strict_trapping:
        for key in ["A_nonempty", "A_subset_X", "A_compact", "F_A_equals_A", "U_contains_A"]:
            conditions[key] = "proved"
        conditions["convergence_for_all_x_in_U"] = "proved"
        report["result"]["candidate_status"] = "not_applicable"
        report["result"]["global_existence_status"] = "proved"
        report["attractors_found"].append(
            {
                "A": "intersection_{n>=0} F^n(closure(U))",
                "U": U_raw,
                "certificate_type": "trapping_region",
                "conditions": conditions,
                "certificate": {
                    "theorem": "strict trapping-region attractor theorem",
                    "F_closure_U_subset_interior_U": certificate.get("F_closure_U_subset_interior_U"),
                    "A_definition": "maximal invariant set inside U",
                },
            }
        )
    finalize_report(report)
    return report


def finalize_candidate_from_conditions(report: dict[str, Any], conditions: dict[str, str]) -> None:
    if "refuted" in conditions.values():
        report["result"]["candidate_status"] = "refuted"
    elif report["query_type"] == "candidate_attractor":
        report["result"]["candidate_status"] = "not_established"


def finalize_report(report: dict[str, Any]) -> None:
    if report["attractors_found"]:
        report["result"]["status"] = "proved"
        if report["result"]["global_existence_status"] == "not_established":
            report["result"]["global_existence_status"] = "proved"
        if report["result"]["candidate_status"] == "refuted":
            report["result"]["meaning"] = (
                "The provided candidate is refuted, but at least one strict mathematical "
                "attractor was proven for the system."
            )
        elif report["result"]["candidate_status"] == "proved":
            report["result"]["meaning"] = "The provided candidate is a strict mathematical attractor."
        else:
            report["result"]["meaning"] = "At least one strict mathematical attractor was proven."
        return

    if (
        report["query_type"] == "candidate_attractor"
        and report["result"]["candidate_status"] == "refuted"
    ):
        report["result"]["status"] = "refuted"
        report["result"]["meaning"] = "The provided candidate violates a necessary attractor condition."
        return
    if (
        report["query_type"] == "existence_any"
        and report["result"]["global_existence_status"] == "refuted"
    ):
        report["result"]["status"] = "refuted"
        report["result"]["meaning"] = "No strict attractor exists for the relevant finite query."
        return
    report["result"]["status"] = "not_established"
    if not report["result"].get("meaning") or report["result"]["meaning"].startswith("No strict"):
        report["result"]["meaning"] = (
            "Available data is insufficient for a strict proof or strict refutation."
        )


class CertificateVerifier:
    def verify(self, data: dict[str, Any], mode: str) -> dict[str, Any]:
        query_type = data.get("query_type", "existence_any") if isinstance(data, dict) else "existence_any"
        report = base_report(query_type)
        report["diagnostics_not_used_as_proof"] = collect_diagnostics_not_used_as_proof(data, mode)
        if not isinstance(data, dict):
            finalize_report(report)
            return report
        if mode == "finite_exhaustive":
            return verify_finite_exhaustive(data, report)
        if mode == "contraction_certificate":
            return verify_contraction_certificate(data, report)
        if mode == "lyapunov_certificate":
            return verify_lyapunov_certificate(data, report)
        if mode == "trapping_region_certificate":
            return verify_trapping_region_certificate(data, report)
        raise ValueError(f"Unsupported mode: {mode}")


def verify(data: dict[str, Any], mode: str) -> dict[str, Any]:
    return CertificateVerifier().verify(data, mode)


def malformed_json_report(path: str, mode: str, error: Exception) -> dict[str, Any]:
    report = base_report("existence_any")
    report["diagnostics_not_used_as_proof"] = {
        "mode_requested": mode,
        "input_path": path,
        "json_error": str(error),
    }
    report["result"]["meaning"] = "Malformed JSON: no proof could be established."
    finalize_report(report)
    return report


def write_report(report: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify strict mathematical attractor existence proofs.")
    parser.add_argument("--mode", required=True, help="Verification mode.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", help="Output report JSON path. Defaults to stdout.")
    args = parser.parse_args(argv)

    if args.mode not in SUPPORTED_MODES:
        print(f"Invalid mode: {args.mode}", file=sys.stderr)
        return 2

    try:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report = malformed_json_report(args.input, args.mode, exc)
        write_report(report, args.output)
        return 0
    except OSError as exc:
        print(f"Could not read input JSON: {exc}", file=sys.stderr)
        return 2

    report = verify(data, args.mode)
    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
