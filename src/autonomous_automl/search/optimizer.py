"""Persistent ask/tell Optuna optimization scoped to one pipeline family."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import optuna
from optuna.trial import TrialState

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    PipelineSpec,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.pipelines import check_pipeline_compatibility, pipeline_fingerprint
from autonomous_automl.utils.errors import ConfigurationError, ResumeError
from autonomous_automl.utils.hashing import sha256_json
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps
from autonomous_automl.utils.seeds import derive_seed


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Runtime link between one Optuna trial and one executable specification."""

    candidate_id: str
    family: str
    model_name: str
    study_name: str
    trial_number: int
    template_fingerprint: str
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec


class OptunaFamilyOptimizer:
    """Coordinate persistent model-specific studies for one logical family."""

    def __init__(
        self,
        family: str,
        templates: Sequence[PipelineSpec],
        profile: DatasetProfile,
        storage_path: str | Path,
        *,
        random_seed: int,
        registry: ModelRegistry | None = None,
        study_prefix: str = "automl",
    ) -> None:
        if not family:
            raise ValueError("family must not be empty")
        if not templates:
            raise ValueError("at least one compatible template is required")
        if not study_prefix:
            raise ValueError("study_prefix must not be empty")

        self.family = family
        self.profile = profile
        self.random_seed = random_seed
        self.registry = registry or ModelRegistry.default()
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage_path.exists() and not self.storage_path.is_file():
            raise ValueError("Optuna storage_path must identify a SQLite file")

        grouped = self._validate_and_group_templates(templates)
        self._templates = grouped
        self._studies: dict[str, optuna.Study] = {}
        storage_url = f"sqlite:///{self.storage_path.as_posix()}"
        for model_name in sorted(grouped):
            sampler_seed = derive_seed(random_seed, "optuna", family, model_name)
            study_name = f"{study_prefix}:{family}:{model_name}"
            study = optuna.create_study(
                storage=storage_url,
                sampler=optuna.samplers.TPESampler(seed=sampler_seed),
                direction="maximize",
                study_name=study_name,
                load_if_exists=True,
            )
            self._validate_study_identity(study, model_name)
            self._studies[model_name] = study

    @property
    def model_names(self) -> tuple[str, ...]:
        """Return stable model names represented by this optimizer."""

        return tuple(sorted(self._studies))

    def study_for_model(self, model_name: str) -> optuna.Study:
        """Expose a model study for diagnostics and persistence checks."""

        try:
            return self._studies[model_name]
        except KeyError as error:
            raise KeyError(f"model {model_name!r} is not part of family {self.family!r}") from error

    def ask(self, fidelity: FidelitySpec) -> SearchCandidate:
        """Ask the least-explored model study for a new compatible candidate."""
        model_name = min(
            self.model_names,
            key=lambda name: (len(self._studies[name].trials), name),
        )
        study = self._studies[model_name]
        deterministic_seed = derive_seed(
            self.random_seed,
            "optuna",
            self.family,
            model_name,
            len(study.get_trials(deepcopy=False)),
        )
        study.sampler = optuna.samplers.TPESampler(seed=deterministic_seed)
        trial = study.ask()
        try:
            templates = self._templates[model_name]
            fingerprints = tuple(templates)
            selected = trial.suggest_categorical("__template_fingerprint", fingerprints)
            template = templates[selected]
            adapter = self.registry.get(model_name)
            suggested = adapter.suggest_params(trial, self.profile, fidelity)
            params = dict(template.model_params)
            params.update(_validated_json_parameters(suggested))
            spec_payload = template.model_dump(mode="python")
            spec_payload["model_params"] = params
            spec_payload["random_seed"] = derive_seed(
                self.random_seed,
                self.family,
                model_name,
                trial.number,
            )
            spec = PipelineSpec.model_validate(spec_payload)
            canonical_json_dumps(spec.to_json_value())
            decision = check_pipeline_compatibility(spec, self.profile, self.registry)
            if not decision.compatible:
                raise ConfigurationError(
                    "suggested PipelineSpec became incompatible: " + "; ".join(decision.reasons)
                )

            candidate_id = sha256_json(
                {
                    "study_name": study.study_name,
                    "trial_number": trial.number,
                    "pipeline_spec": spec.to_json_value(),
                    "fidelity": fidelity.to_json_value(),
                }
            )
            trial.set_user_attr("candidate_id", candidate_id)
            trial.set_user_attr("pipeline_fingerprint", pipeline_fingerprint(spec))
            trial.set_user_attr("fidelity_level", fidelity.level)
            return SearchCandidate(
                candidate_id=candidate_id,
                family=self.family,
                model_name=model_name,
                study_name=study.study_name,
                trial_number=trial.number,
                template_fingerprint=selected,
                pipeline_spec=spec,
                fidelity=fidelity,
            )
        except Exception:
            study.tell(trial, state=TrialState.FAIL, skip_if_finished=True)
            raise

    def tell(self, candidate: SearchCandidate, result: TrialResult) -> None:
        """Finish an Optuna trial, idempotently accepting an identical replay."""
        study = self._study_for_candidate(candidate)
        self._validate_result(candidate, result)
        expected_state = _trial_state(result.status)
        frozen = _find_trial(study, candidate.trial_number)
        if frozen.state.is_finished():
            _validate_finished_trial(frozen, expected_state, result)
            return

        value = result.mean_score if expected_state is TrialState.COMPLETE else None
        study.tell(
            candidate.trial_number,
            value,
            state=expected_state,
            skip_if_finished=True,
        )

    def reconcile(
        self,
        completed: Iterable[tuple[SearchCandidate, TrialResult]],
    ) -> None:
        """Replay stored terminal results without double-telling finished trials."""
        for candidate, result in completed:
            self.tell(candidate, result)

    def reconcile_results(self, results: Iterable[TrialResult]) -> None:
        """Recover ask/tell after a crash using business results as authority."""
        for result in results:
            candidate = self._candidate_for_result(result)
            self.tell(candidate, result)

    def candidate_by_id(
        self,
        candidate_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
    ) -> SearchCandidate | None:
        """Recover a persisted in-flight candidate from its Optuna study."""
        if not candidate_id.strip():
            raise ValueError("candidate_id must not be blank")
        matches: list[SearchCandidate] = []
        for model_name, study in self._studies.items():
            for trial in study.get_trials(deepcopy=False):
                if trial.user_attrs.get("candidate_id") != candidate_id:
                    continue
                template = trial.params.get("__template_fingerprint")
                if not isinstance(template, str):
                    raise ResumeError("persisted Optuna trial lost its template identity")
                expected_id = sha256_json(
                    {
                        "study_name": study.study_name,
                        "trial_number": trial.number,
                        "pipeline_spec": pipeline_spec.to_json_value(),
                        "fidelity": fidelity.to_json_value(),
                    }
                )
                if expected_id != candidate_id:
                    raise ResumeError("persisted Optuna candidate identity is inconsistent")
                if pipeline_spec.family != self.family or pipeline_spec.model_name != model_name:
                    raise ResumeError("persisted Optuna candidate specification is inconsistent")
                if trial.user_attrs.get("pipeline_fingerprint") != pipeline_fingerprint(
                    pipeline_spec
                ):
                    raise ResumeError("persisted Optuna candidate fingerprint is inconsistent")
                if trial.user_attrs.get("fidelity_level") != fidelity.level:
                    raise ResumeError("persisted Optuna candidate fidelity is inconsistent")
                matches.append(
                    SearchCandidate(
                        candidate_id=candidate_id,
                        family=self.family,
                        model_name=model_name,
                        study_name=study.study_name,
                        trial_number=trial.number,
                        template_fingerprint=template,
                        pipeline_spec=pipeline_spec,
                        fidelity=fidelity,
                    )
                )
        if len(matches) > 1:
            raise ResumeError("candidate id occurs in more than one Optuna study")
        return matches[0] if matches else None

    def _validate_and_group_templates(
        self,
        templates: Sequence[PipelineSpec],
    ) -> dict[str, dict[str, PipelineSpec]]:
        grouped: dict[str, dict[str, PipelineSpec]] = {}
        for template in templates:
            if template.family != self.family:
                raise ValueError("all templates must belong to the optimizer family")
            decision = check_pipeline_compatibility(template, self.profile, self.registry)
            if not decision.compatible:
                raise ValueError("incompatible optimizer template: " + "; ".join(decision.reasons))
            fingerprint = pipeline_fingerprint(template)
            model_templates = grouped.setdefault(template.model_name, {})
            if fingerprint in model_templates:
                raise ValueError("optimizer templates must have unique fingerprints")
            model_templates[fingerprint] = template
        return grouped

    def _validate_study_identity(self, study: optuna.Study, model_name: str) -> None:
        current_templates = self._templates[model_name]
        expected: dict[str, JsonValue] = {
            "family": self.family,
            "model_name": model_name,
            "dataset_hash": self.profile.dataset_hash,
            "random_seed": self.random_seed,
            "template_fingerprints": list(current_templates),
        }
        identity = study.user_attrs.get("optimizer_identity")
        if identity is None:
            study.set_user_attr("optimizer_identity", expected)
            return
        if not isinstance(identity, dict):
            raise ResumeError("persisted Optuna study identity is malformed")

        stable_keys = ("family", "model_name", "dataset_hash", "random_seed")
        if any(identity.get(key) != expected[key] for key in stable_keys):
            raise ResumeError("persisted Optuna study identity does not match this optimizer")

        persisted_fingerprints = identity.get("template_fingerprints")
        if (
            not isinstance(persisted_fingerprints, list)
            or not persisted_fingerprints
            or any(not isinstance(value, str) for value in persisted_fingerprints)
            or len(persisted_fingerprints) != len(set(persisted_fingerprints))
        ):
            raise ResumeError("persisted Optuna template identity is malformed")
        if any(fingerprint not in current_templates for fingerprint in persisted_fingerprints):
            raise ResumeError("persisted Optuna template is unavailable in this version")

        # Additive template evolution is safe only for new runs. A resumed study keeps the
        # exact ordered template universe recorded when it started, preserving both Optuna's
        # categorical distribution and deterministic candidate identities.
        self._templates[model_name] = {
            fingerprint: current_templates[fingerprint] for fingerprint in persisted_fingerprints
        }

    def _study_for_candidate(self, candidate: SearchCandidate) -> optuna.Study:
        if candidate.family != self.family:
            raise ValueError("candidate belongs to a different optimizer family")
        study = self.study_for_model(candidate.model_name)
        if candidate.study_name != study.study_name:
            raise ValueError("candidate study name does not match the model study")
        return study

    def _candidate_for_result(self, result: TrialResult) -> SearchCandidate:
        candidate = self.candidate_by_id(
            result.trial_id,
            result.pipeline_spec,
            result.fidelity,
        )
        if candidate is None:
            raise ResumeError("persisted TrialResult has no matching Optuna trial")
        return candidate

    @staticmethod
    def _validate_result(candidate: SearchCandidate, result: TrialResult) -> None:
        if result.trial_id != candidate.candidate_id:
            raise ValueError("TrialResult.trial_id must match SearchCandidate.candidate_id")
        if result.family != candidate.family:
            raise ValueError("TrialResult.family must match SearchCandidate.family")
        if result.pipeline_spec != candidate.pipeline_spec:
            raise ValueError("TrialResult.pipeline_spec must match the candidate")
        if result.fidelity != candidate.fidelity:
            raise ValueError("TrialResult.fidelity must match the candidate")


def _validated_json_parameters(parameters: dict[str, object]) -> dict[str, JsonValue]:
    if any(not isinstance(name, str) for name in parameters):
        raise ConfigurationError("model adapter parameter names must be strings")
    try:
        canonical_json_dumps(cast(dict[str, JsonValue], parameters))
    except (TypeError, ValueError) as error:
        raise ConfigurationError("model adapter parameters must be finite JSON values") from error
    return cast(dict[str, JsonValue], parameters)


def _trial_state(status: TrialStatus) -> TrialState:
    mapping = {
        TrialStatus.COMPLETED: TrialState.COMPLETE,
        TrialStatus.FAILED: TrialState.FAIL,
        TrialStatus.PRUNED: TrialState.PRUNED,
    }
    try:
        return mapping[status]
    except KeyError as error:
        raise ValueError(f"unsupported Optuna terminal status: {status.value}") from error


def _find_trial(study: optuna.Study, trial_number: int) -> optuna.trial.FrozenTrial:
    for trial in study.get_trials(deepcopy=False):
        if trial.number == trial_number:
            return trial
    raise ResumeError("Optuna trial referenced by the candidate is missing")


def _validate_finished_trial(
    trial: optuna.trial.FrozenTrial,
    expected_state: TrialState,
    result: TrialResult,
) -> None:
    if trial.state is not expected_state:
        raise ResumeError("stored Optuna state conflicts with the persisted TrialResult")
    if expected_state is TrialState.COMPLETE and trial.value != result.mean_score:
        raise ResumeError("stored Optuna value conflicts with the persisted TrialResult")


__all__ = ["OptunaFamilyOptimizer", "SearchCandidate"]
