"""The pinned versions agree everywhere they are written down (offline: only files are read).

The pipeline image trains the model and the API image loads it, so TensorFlow, Keras and the
packages that both use must have the same version in both; pyproject.toml must accept exactly what
the requirements files install (``pip install -e .`` then installs nothing a second time); the
MLflow and Prefect servers must run the versions of the clients."""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = {
    "pipeline": ROOT / "services" / "pipeline" / "requirements.txt",
    "api": ROOT / "services" / "api" / "requirements.txt",
    "dev": ROOT / "requirements-dev.txt",
}


def pins(path: Path) -> dict[str, str]:
    """``name -> version`` of the ``name==version`` lines (``-r`` includes are not followed)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        req = Requirement(line)
        (spec,) = req.specifier  # every requirement is pinned to one version
        assert spec.operator == "==", f"{path.name}: {line} is not pinned"
        out[canonicalize_name(req.name)] = spec.version
    return out


@pytest.fixture(scope="module")
def pinned():
    return {name: pins(path) for name, path in REQUIREMENTS.items()}


def test_shared_packages_have_the_same_pin_in_both_images(pinned):
    pipeline, api = pinned["pipeline"], pinned["api"]
    shared = set(pipeline) & set(api)
    assert {"tensorflow-cpu", "keras", "numpy", "pandas", "scikit-learn"} <= shared
    assert {k: (pipeline[k], api[k]) for k in shared if pipeline[k] != api[k]} == {}
    # the API uses mlflow-skinny, the pipeline the full mlflow: one release
    assert api["mlflow-skinny"] == pipeline["mlflow"]
    assert set(pinned["dev"]).isdisjoint(pipeline) and set(pinned["dev"]).isdisjoint(api)


def test_pyproject_accepts_exactly_what_the_requirements_install(pinned):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    installed = {**pinned["pipeline"], **pinned["api"], **pinned["dev"]}
    installed["mlflow-skinny"] = pinned["api"]["mlflow-skinny"]
    declared = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        declared += [r for r in extra if not r.startswith(project["name"])]
    seen = set()
    for line in declared:
        req = Requirement(line)
        name = canonicalize_name(req.name)
        if name == "tensorflow":  # the full wheel on platforms without tensorflow-cpu
            assert req.specifier.contains(installed["tensorflow-cpu"]), line
            continue
        assert name in installed, f"{line} is not pinned in a requirements file"
        assert req.specifier.contains(installed[name]), f"{line} vs {name}=={installed[name]}"
        seen.add(name)
    assert seen == set(installed), f"pinned but not declared: {set(installed) - seen}"


def test_servers_run_the_client_versions(pinned):
    mlflow_image = (ROOT / "services" / "mlflow" / "Dockerfile").read_text(encoding="utf-8")
    assert f"mlflow=={pinned['pipeline']['mlflow']}" in mlflow_image
    assert f"psycopg[binary]=={pinned['pipeline']['psycopg']}" in mlflow_image
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    (prefect_tag,) = re.findall(r"image:\s*prefecthq/prefect:([\w.]+)-python", compose)
    assert prefect_tag == pinned["pipeline"]["prefect"]
