"""
Unit & Manifest Validation Tests for Stage 6: Kubernetes Orchestration & HPA
=============================================================================
Verifies Kubernetes manifests for syntax, securityContext, zero-downtime
RollingUpdate strategies, liveness/readiness/startup probes, HPA autoscaling
targets, and NGINX canary traffic splitting annotations.
"""

from pathlib import Path
import pytest
import yaml

from mlops_platform.config.settings import settings


@pytest.fixture
def k8s_dir():
    k8s_path = settings.PROJECT_ROOT / "k8s"
    assert k8s_path.exists(), "Missing 'k8s' directory in project root!"
    return k8s_path


def load_all_yaml_docs(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))


class TestKubernetesManifestsStructure:
    """Verifies all required Kubernetes resources exist and parse without syntax errors."""

    def test_required_manifest_files_exist(self, k8s_dir):
        required_files = [
            "namespace.yaml",
            "configmap.yaml",
            "deployment.yaml",
            "canary-deployment.yaml",
            "service.yaml",
            "hpa.yaml",
            "ingress.yaml",
        ]
        for fname in required_files:
            target = k8s_dir / fname
            assert target.exists(), f"Missing required K8s manifest: {fname}"

    def test_namespace_configuration(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "namespace.yaml")
        assert len(docs) == 1
        ns = docs[0]
        assert ns["kind"] == "Namespace"
        assert ns["metadata"]["name"] == "mlops-production"
        assert ns["metadata"]["labels"]["environment"] == "production"

    def test_configmap_data(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "configmap.yaml")
        cm = docs[0]
        assert cm["kind"] == "ConfigMap"
        assert cm["metadata"]["name"] == "mlops-inference-config"
        data = cm["data"]
        assert data["APP_ENV"] == "production"
        assert data["ENABLE_ONNX_RUNTIME"] == "true"
        assert data["PROMETHEUS_METRICS_ENABLED"] == "true"


class TestDeploymentArchitectureAndSecurity:
    """Verifies deployment strategies, non-root securityContext, and health probes."""

    def test_production_deployment_strategy_and_replicas(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "deployment.yaml")
        dep = docs[0]
        assert dep["kind"] == "Deployment"
        assert dep["metadata"]["name"] == "mlops-inference-deployment"
        assert dep["spec"]["replicas"] >= 3, "Production HA requires >= 3 replicas!"

        # Zero-Downtime RollingUpdate strategy
        strategy = dep["spec"]["strategy"]
        assert strategy["type"] == "RollingUpdate"
        rolling = strategy["rollingUpdate"]
        assert rolling["maxUnavailable"] == 0, "maxUnavailable must be 0 for zero-downtime rollouts!"

    def test_pod_and_container_security_posture(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "deployment.yaml")
        pod_spec = docs[0]["spec"]["template"]["spec"]

        # Pod-level security
        pod_sec = pod_spec.get("securityContext", {})
        assert pod_sec.get("runAsNonRoot") is True, "Pod must enforce runAsNonRoot: true!"
        assert pod_sec.get("runAsUser") == 10001, "Pod must run under unprivileged UID 10001!"
        assert pod_sec.get("runAsGroup") == 10001

        # Container-level security
        container = pod_spec["containers"][0]
        cont_sec = container.get("securityContext", {})
        assert cont_sec.get("allowPrivilegeEscalation") is False, "Container must disallow privilege escalation!"
        assert "ALL" in cont_sec.get("capabilities", {}).get("drop", []), "Container must drop ALL capabilities!"

    def test_operational_health_probes(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "deployment.yaml")
        container = docs[0]["spec"]["template"]["spec"]["containers"][0]

        # Liveness Probe
        assert "livenessProbe" in container, "Missing livenessProbe in deployment!"
        liveness = container["livenessProbe"]["httpGet"]
        assert liveness["path"] == "/health/live"
        assert liveness["port"] == 8000

        # Readiness Probe
        assert "readinessProbe" in container, "Missing readinessProbe in deployment!"
        readiness = container["readinessProbe"]["httpGet"]
        assert readiness["path"] == "/health/ready"
        assert readiness["port"] == 8000

        # Startup Probe
        assert "startupProbe" in container, "Missing startupProbe in deployment!"
        startup = container["startupProbe"]["httpGet"]
        assert startup["path"] == "/health/live"

    def test_resource_requests_and_limits(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "deployment.yaml")
        container = docs[0]["spec"]["template"]["spec"]["containers"][0]
        resources = container.get("resources", {})

        assert "requests" in resources and "limits" in resources
        assert resources["requests"]["cpu"] == "250m"
        assert resources["requests"]["memory"] == "512Mi"
        assert resources["limits"]["cpu"] == "1000m"
        assert resources["limits"]["memory"] == "1024Mi"

    def test_canary_deployment_isolated_track(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "canary-deployment.yaml")
        canary = docs[0]
        assert canary["metadata"]["name"] == "mlops-inference-canary"
        assert canary["spec"]["selector"]["matchLabels"]["track"] == "canary"
        assert canary["spec"]["template"]["metadata"]["labels"]["track"] == "canary"


class TestAutoscalingAndNetworking:
    """Verifies HorizontalPodAutoscaler, Services, and Canary Ingress traffic routing."""

    def test_hpa_configuration(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "hpa.yaml")
        hpa = docs[0]
        assert hpa["kind"] == "HorizontalPodAutoscaler"
        assert hpa["spec"]["scaleTargetRef"]["name"] == "mlops-inference-deployment"
        assert hpa["spec"]["minReplicas"] == 2
        assert hpa["spec"]["maxReplicas"] == 10

        metrics = {m["resource"]["name"]: m["resource"]["target"]["averageUtilization"] for m in hpa["spec"]["metrics"]}
        assert metrics["cpu"] == 70, "Target CPU utilization must be 70%!"
        assert metrics["memory"] == 80, "Target memory utilization must be 80%!"

        # Scale down stabilization to avoid flapping
        assert hpa["spec"]["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == 300

    def test_service_targets_and_ports(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "service.yaml")
        svc_map = {doc["metadata"]["name"]: doc for doc in docs}

        assert "mlops-inference-service" in svc_map
        assert "mlops-canary-service" in svc_map
        assert "mlops-unified-service" in svc_map

        stable_svc = svc_map["mlops-inference-service"]
        assert stable_svc["spec"]["selector"]["track"] == "stable"
        assert stable_svc["spec"]["ports"][0]["port"] == 80
        assert stable_svc["spec"]["ports"][0]["targetPort"] == 8000

        canary_svc = svc_map["mlops-canary-service"]
        assert canary_svc["spec"]["selector"]["track"] == "canary"

    def test_canary_ingress_routing(self, k8s_dir):
        docs = load_all_yaml_docs(k8s_dir / "ingress.yaml")
        ingress_map = {doc["metadata"]["name"]: doc for doc in docs}

        assert "mlops-production-ingress" in ingress_map
        assert "mlops-canary-ingress" in ingress_map

        canary_ing = ingress_map["mlops-canary-ingress"]
        annotations = canary_ing["metadata"]["annotations"]

        # Advanced Enhancement #4: Canary & Shadow Traffic Routing
        assert annotations.get("nginx.ingress.kubernetes.io/canary") == "true"
        assert annotations.get("nginx.ingress.kubernetes.io/canary-weight") == "10"
        assert annotations.get("nginx.ingress.kubernetes.io/canary-by-header") == "X-Model-Experiment"
        assert annotations.get("nginx.ingress.kubernetes.io/canary-by-header-value") == "canary"
