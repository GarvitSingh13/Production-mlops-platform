"""
Unit & Integration Tests for Stage 5: Production Docker Containerization
========================================================================
Verifies multi-stage build structure, non-root security compliance,
healthcheck declarations, .dockerignore security coverage, and docker-compose.
"""

import pytest
import yaml

from mlops_platform.config.settings import settings


@pytest.fixture
def project_root():
    return settings.PROJECT_ROOT


class TestDockerfileStandards:
    """Verifies Dockerfile conforms to production container engineering best practices."""

    def test_dockerfile_exists(self, project_root):
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Missing Dockerfile in project root!"

    def test_multistage_build_structure(self, project_root):
        content = (project_root / "Dockerfile").read_text(encoding="utf-8")
        from_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("FROM")]

        # Must have at least 2 distinct stages
        assert len(from_lines) >= 2, f"Expected multi-stage build with >= 2 FROM lines, found: {from_lines}"
        assert any("AS builder" in line for line in from_lines), "Missing 'builder' stage in Dockerfile!"
        assert any("AS runtime" in line for line in from_lines), "Missing 'runtime' stage in Dockerfile!"

    def test_security_hardening_non_root_user(self, project_root):
        content = (project_root / "Dockerfile").read_text(encoding="utf-8")

        # Must create unprivileged user and switch via USER directive
        assert "useradd" in content or "adduser" in content, "Dockerfile must create an unprivileged user!"
        assert "USER appuser" in content, "Dockerfile must execute under non-root 'USER appuser'!"

    def test_operational_directives(self, project_root):
        content = (project_root / "Dockerfile").read_text(encoding="utf-8")

        assert "EXPOSE 8000" in content, "Dockerfile must declare EXPOSE 8000!"
        assert "HEALTHCHECK" in content, "Dockerfile must declare HEALTHCHECK for container probes!"
        assert "uvicorn" in content, "Dockerfile entrypoint must launch uvicorn ASGI server!"


class TestDockerIgnoreAndCompose:
    """Verifies .dockerignore security coverage and docker-compose orchestration."""

    def test_dockerignore_rules(self, project_root):
        ignore_path = project_root / ".dockerignore"
        assert ignore_path.exists(), "Missing .dockerignore in project root!"

        rules = ignore_path.read_text(encoding="utf-8")

        # Must ignore sensitive, private, and bulky directories
        assert "PROJECT_OVERVIEW.md" in rules, ".dockerignore must exclude private PROJECT_OVERVIEW.md!"
        assert ".git" in rules, ".dockerignore must exclude .git repository data!"
        assert ".venv" in rules, ".dockerignore must exclude local virtual environment!"
        assert "__pycache__" in rules, ".dockerignore must exclude Python bytecode cache!"
        assert "data/raw" in rules, ".dockerignore must exclude raw data to keep images minimal!"

    def test_docker_compose_syntax_and_services(self, project_root):
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "Missing docker-compose.yml in project root!"

        with open(compose_path, "r", encoding="utf-8") as f:
            compose_cfg = yaml.safe_load(f)

        assert "services" in compose_cfg
        assert "inference-service" in compose_cfg["services"]

        service = compose_cfg["services"]["inference-service"]
        assert "8000:8000" in service.get("ports", [])
        assert "healthcheck" in service
        assert "deploy" in service
