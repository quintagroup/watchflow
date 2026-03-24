"""
Risk assessment signal evaluators.

Each evaluator is pure/synchronous and returns a list of RiskSignal instances,
one per triggered condition. Returns [] if nothing noteworthy was found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agents import get_agent
from src.core.models import EventType, Severity, Violation
from src.rules.loaders.github_loader import github_rule_loader

if TYPE_CHECKING:
    from src.rules.models import Rule


@dataclass
class RiskSignal:
    category: str
    severity: Severity
    description: str


@dataclass
class RiskAssessmentResult:
    level: Severity
    signals: list[RiskSignal] = field(default_factory=list)


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


# (regex, severity, label_for_description)
_CRITICAL_PATH_PATTERNS: list[tuple[str, Severity, str]] = [
    (r"(^|/)auth/", Severity.HIGH, "auth"),
    (r"(^|/)oauth/", Severity.HIGH, "auth"),
    (r"(^|/)authentication/", Severity.HIGH, "auth"),
    (r"(^|/)login[./]", Severity.HIGH, "auth"),
    (r"(^|/)payments?/", Severity.HIGH, "payments"),
    (r"(^|/)billing/", Severity.HIGH, "payments"),
    (r"(^|/)migrations?/", Severity.HIGH, "db-migration"),
    (r"\.sql$", Severity.HIGH, "db-migration"),
]

_SECURITY_PATTERNS: list[tuple[str, Severity, str]] = [
    (r"(^|/)\.env", Severity.CRITICAL, "env-secrets"),
    (r"(^|/)secrets?/", Severity.CRITICAL, "secrets"),
    (r"\.pem$", Severity.CRITICAL, "private-key"),
    (r"\.key$", Severity.CRITICAL, "private-key"),
    (r"^\.github/workflows/", Severity.HIGH, "ci-cd"),
    (r"^\.circleci/", Severity.HIGH, "ci-cd"),
    (r"^Jenkinsfile$", Severity.HIGH, "ci-cd"),
    (r"^\.travis\.yml$", Severity.HIGH, "ci-cd"),
    (r"^Dockerfile[^/]*$", Severity.MEDIUM, "dockerfile"),
    (r"^docker-compose[^/]*$", Severity.MEDIUM, "dockerfile"),
    (r"\.tf$", Severity.MEDIUM, "infra"),
    (r"(^|/)terraform/", Severity.MEDIUM, "infra"),
]

_BREAKING_PATTERNS: list[tuple[str, Severity, str]] = [
    (r".proto", Severity.MEDIUM, "protobuf"),
    (r"schema.graphql", Severity.MEDIUM, "graphql-schema"),
    (r"openapi.yaml", Severity.MEDIUM, "openapi"),
    (r"swagger", Severity.MEDIUM, "swagger"),
]

_DEP_FILE_PATTERNS: list[str] = [
    r"(^|/)package\.json$",
    r"(^|/)requirements[^/]*\.txt$",
    r"(^|/)go\.mod$",
    r"(^|/)Cargo\.toml$",
    r"(^|/)pom\.xml$",
    r"(^|/)build\.gradle$",
    r"(^|/)Pipfile$",
]


def _match_patterns(
    filepath: str,
    patterns: list[tuple[str, Severity, str]],
    category: str,
) -> list[RiskSignal]:
    """Return one RiskSignal per pattern group that matches the filepath (deduped by label)."""
    signals = []
    seen_labels: set[str] = set()
    for pattern, severity, label in patterns:
        if re.search(pattern, filepath, re.IGNORECASE):
            key = f"{label}:{filepath}"
            if key not in seen_labels:
                seen_labels.add(key)
                signals.append(
                    RiskSignal(
                        category=category,
                        severity=severity,
                        description=f"`{filepath}` matches {label} pattern",
                    )
                )
    return signals


async def _evaluate_rules(
    repo: str,
    installation_id: int,
    pr_data: dict[str, Any],
    pr_files: list[dict[str, Any]],
) -> list[Violation]:
    """Run rule conditions against the PR and return violations."""
    try:
        rules: list[Rule] = await github_rule_loader.get_rules(repo, installation_id)
    except Exception:
        return [], []

    # Filter pull request rules
    pr_rules = [rule for rule in rules if EventType.PULL_REQUEST in rule.event_types]

    # Use the same format as PullRequestProcessor
    formatted_rules = []

    for rule in pr_rules:
        # Convert Rule object to dict format
        rule_dict = {
            "description": rule.description,
            "enabled": rule.enabled,
            "severity": rule.severity.value if hasattr(rule.severity, "value") else rule.severity,
            "event_types": [et.value if hasattr(et, "value") else et for et in rule.event_types],
            "parameters": rule.parameters if hasattr(rule, "parameters") else {},
        }

        formatted_rules.append(rule_dict)

    # Prepare event data in the format expected by the agentic analysis
    event_data = {
        "pull_request_details": pr_data,
        "files": pr_files,
        "repository": {"full_name": repo},
        "installation": {"id": installation_id},
    }

    engine_agent = get_agent("engine")
    result = await engine_agent.execute(
        event_type="pull_request",
        event_data=event_data,
        rules=formatted_rules,
    )

    return pr_rules, result.data.get("violations", [])


def evaluate_size(pr_data: dict[str, Any], files: list[dict[str, Any]]) -> list[RiskSignal]:
    """Produce signals for large PR size across three dimensions."""
    signals: list[RiskSignal] = []

    file_count = len(files)
    if file_count > 50:
        signals.append(RiskSignal("size", Severity.HIGH, f"{file_count} files changed"))
    elif file_count > 30:
        signals.append(RiskSignal("size", Severity.MEDIUM, f"{file_count} files changed"))

    loc = int(pr_data.get("additions") or 0) + int(pr_data.get("deletions") or 0)
    if loc > 3000:
        signals.append(RiskSignal("size", Severity.HIGH, f"{loc:,} lines changed"))
    elif loc > 1000:
        signals.append(RiskSignal("size", Severity.MEDIUM, f"{loc:,} lines changed"))

    commits = int(pr_data.get("commits") or 0)
    if commits > 30:
        signals.append(RiskSignal("size", Severity.HIGH, f"{commits} commits"))
    elif commits > 15:
        signals.append(RiskSignal("size", Severity.MEDIUM, f"{commits} commits"))

    return signals


def evaluate_critical_path(files: list[dict[str, Any]]) -> list[RiskSignal]:
    """One signal per file matching a critical business path pattern."""
    signals: list[RiskSignal] = []
    for f in files:
        filepath = f.get("filename", "")
        signals.extend(_match_patterns(filepath, _CRITICAL_PATH_PATTERNS, "critical-path"))
    return signals


def evaluate_test_coverage(files: list[dict[str, Any]]) -> list[RiskSignal]:
    """Detect removed test files and source files added without test counterparts."""
    # Find test and non-test source files
    source_files = []
    test_files = []

    # Default test pattern looks for tests/ directory or files ending in test.py/test.ts etc
    compiled_pattern = re.compile(r"(^tests?/|test\.[a-zA-Z]+$|_test\.[a-zA-Z]+$)")

    for f in files:
        filename = f.get("filename", "")
        if not filename:
            continue

        # Ignore documentation and config files
        if filename.endswith((".md", ".txt", ".yaml", ".json")):
            continue

        if compiled_pattern.search(filename):
            test_files.append(filename)
        else:
            source_files.append(filename)

    # If source files were modified but no test files were modified
    if source_files and not test_files:
        return [
            RiskSignal(
                "test-coverage",
                Severity.MEDIUM,
                "Source files were modified without corresponding test changes.",
            )
        ]
    return []


def evaluate_dependency_changes(files: list[dict[str, Any]]) -> list[RiskSignal]:
    """One signal per dependency manifest modified."""
    signals: list[RiskSignal] = []
    for f in files:
        filepath = f.get("filename", "")
        for pattern in _DEP_FILE_PATTERNS:
            if re.search(pattern, filepath, re.IGNORECASE):
                signals.append(RiskSignal("dependency", Severity.MEDIUM, f"Dependency file modified: `{filepath}`"))
                break
    return signals


def evaluate_contributor_history(pr_data: dict[str, Any]) -> list[RiskSignal]:
    """Signal based on the author's association to the repository."""
    association = (pr_data.get("author_association") or "").upper()
    if association in ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"):
        return [RiskSignal("contributor", Severity.HIGH, "Author is a first-time contributor")]
    return []


def evaluate_reverts(pr_data: dict[str, Any]) -> list[RiskSignal]:
    """Signal if the PR appears to be a revert."""
    title = (pr_data.get("title") or "").lower()
    branch = (pr_data.get("head", {}).get("ref") or "").lower()
    if "revert" in title:
        return [RiskSignal("revert", Severity.HIGH, f"PR title contains 'revert': `{pr_data.get('title')}`")]
    if "revert" in branch:
        return [RiskSignal("revert", Severity.HIGH, f"Branch name contains 'revert': `{branch}`")]
    return []


def evaluate_security_sensitive(files: list[dict[str, Any]]) -> list[RiskSignal]:
    """One signal per file matching security-sensitive infrastructure patterns."""
    signals: list[RiskSignal] = []
    for f in files:
        filepath = f.get("filename", "")
        signals.extend(_match_patterns(filepath, _SECURITY_PATTERNS, "security"))
    return signals


def evaluate_breaking_changes(pr_data: dict[str, Any], files: list[dict[str, Any]]) -> list[RiskSignal]:
    """One signal per file that could indicate a breaking change."""
    signals: list[RiskSignal] = []
    for f in files:
        filepath = f.get("filename", "")
        signals.extend(_match_patterns(filepath, _BREAKING_PATTERNS, "breaking-change"))
    return signals


def evaluate_rule_matches(violations: list[Violation]) -> list[RiskSignal]:
    """One signal per rule violation, severity mapped from violation severity."""
    signals: list[RiskSignal] = [
        RiskSignal(
            "rule-violation",
            v.severity,
            f"Rule violation: {v.rule_description}",
        )
        for v in violations
        if v.severity != Severity.INFO
    ]
    return signals


def compute_risk(signals: list[RiskSignal]) -> RiskAssessmentResult:
    """Overall risk level = maximum severity across all triggered signals."""
    if not signals:
        risk_level = Severity.LOW
    else:
        risk_level = max((s.severity for s in signals), key=lambda sev: _SEVERITY_ORDER[sev])
    return RiskAssessmentResult(level=risk_level, signals=signals)


async def generate_risk_assessment(
    repo: str,
    installation_id: int,
    pr_data: dict[str, Any],
    pr_files: list[dict[str, Any]],
):
    rules, violations = await _evaluate_rules(repo, installation_id, pr_data, pr_files)

    signals = []
    signals.extend(evaluate_rule_matches(violations))
    signals.extend(evaluate_size(pr_data, pr_files))

    if not any(rule.parameters.get("critical_owners", []) for rule in rules):
        signals.extend(evaluate_critical_path(pr_files))

    if not any(rule.parameters.get("require_tests", False) for rule in rules):
        signals.extend(evaluate_test_coverage(pr_files))

    signals.extend(evaluate_dependency_changes(pr_files))
    signals.extend(evaluate_contributor_history(pr_data))
    signals.extend(evaluate_reverts(pr_data))

    if not any(rule.parameters.get("security_patterns", False) for rule in rules):
        signals.extend(evaluate_security_sensitive(pr_files))

    signals.extend(evaluate_breaking_changes(pr_data, pr_files))

    return compute_risk(signals)
