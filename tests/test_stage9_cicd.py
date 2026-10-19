"""
Unit & Integration Tests for Stage 9: CI/CD Pipeline & Master Orchestration
============================================================================
Verifies GitHub Actions workflow syntax, pipeline job sequencing, trigger events,
caching policies, master verification runner, and repository completeness.
"""

import pytest
import yaml

from mlops_platform.config.settings import settings


@pytest.fixture
def workflow_yaml():
    wf_path = settings.PROJECT_ROOT / ".github" / "workflows" / "ci-cd.yaml"
    assert wf_path.exists(), "Missing .github/workflows/ci-cd.yaml!"

    with open(wf_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestGitHubActionsWorkflow:
    """Verifies GitHub Actions workflow conforms to enterprise CI/CD standards."""

    def test_workflow_triggers_and_concurrency(self, workflow_yaml):
        triggers = workflow_yaml.get("on") or workflow_yaml.get(True, {})
        assert "push" in triggers, "Workflow must trigger on git push!"
        assert "pull_request" in triggers, "Workflow must trigger on pull request!"
        assert "workflow_dispatch" in triggers, "Workflow must support manual workflow_dispatch!"

        push_branches = triggers["push"].get("branches", [])
        assert "main" in push_branches, "Workflow must monitor the main branch!"

        assert "concurrency" in workflow_yaml, "Workflow must configure concurrency cancel-in-progress!"

    def test_workflow_job_definitions(self, workflow_yaml):
        jobs = workflow_yaml.get("jobs", {})

        expected_jobs = [
            "lint-and-quality",
            "data-and-training-pipeline",
            "serving-and-monitoring-pipeline",
            "container-and-k8s-audit",
            "master-e2e-verification",
        ]
        for j in expected_jobs:
            assert j in jobs, f"Missing job in CI/CD pipeline: {j}"

    def test_workflow_job_dependencies(self, workflow_yaml):
        jobs = workflow_yaml["jobs"]

        # Verify DAG dependency ordering
        assert jobs["data-and-training-pipeline"].get("needs") == "lint-and-quality"
        assert jobs["serving-and-monitoring-pipeline"].get("needs") == "data-and-training-pipeline"
        assert jobs["container-and-k8s-audit"].get("needs") == "serving-and-monitoring-pipeline"
        assert jobs["master-e2e-verification"].get("needs") == "container-and-k8s-audit"


class TestMasterOrchestratorStructure:
    """Verifies run_all_stages.py master test harness structure."""

    def test_run_all_stages_exists(self):
        orchestrator = settings.PROJECT_ROOT / "run_all_stages.py"
        assert orchestrator.exists(), "Missing run_all_stages.py in project root!"

    def test_run_all_stages_covers_all_test_suites(self):
        content = (settings.PROJECT_ROOT / "run_all_stages.py").read_text(encoding="utf-8")

        for stage_num in range(1, 9):
            pattern = f"test_stage{stage_num}_"
            assert pattern in content, f"run_all_stages.py must include stage {stage_num} test suite!"


class TestRepositoryCompleteness:
    """Verifies that all 8 previous stage runners and test suites exist."""

    def test_all_stage_runners_exist(self):
        for stage_num in range(1, 9):
            runner = settings.PROJECT_ROOT / f"run_stage{stage_num}.py"
            assert runner.exists(), f"Missing runner for stage {stage_num}: run_stage{stage_num}.py"

    def test_gitignore_security_rules(self):
        gitignore_path = settings.PROJECT_ROOT / ".gitignore"
        assert gitignore_path.exists(), "Missing .gitignore in project root!"
        content = gitignore_path.read_text(encoding="utf-8")

        assert "PROJECT_OVERVIEW.md" in content, ".gitignore must exclude private PROJECT_OVERVIEW.md!"
        assert ".venv" in content, ".gitignore must exclude .venv!"
