"""
Consistency checks: verify that documentation matches the actual package state.

These tests fail when code changes outpace documentation, forcing the
README and other public files to stay current.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import apastats

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"


class TestVersionConsistency:
    """Version must match across pyproject.toml, __init__.py, and CITATION.cff."""

    def _extract_version_pyproject(self):
        text = PYPROJECT.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else None

    def _extract_version_citation(self):
        text = CITATION.read_text(encoding="utf-8")
        # Match "version: X.Y.Z" but not "cff-version: X.Y.Z"
        matches = re.findall(r"^version:\s*(.+)", text, re.MULTILINE)
        return matches[0].strip() if matches else None

    def test_init_matches_pyproject(self):
        assert apastats.__version__ == self._extract_version_pyproject(), (
            f"__init__.py says {apastats.__version__}, "
            f"pyproject.toml says {self._extract_version_pyproject()}"
        )

    def test_citation_matches_pyproject(self):
        pyproject_ver = self._extract_version_pyproject()
        citation_ver = self._extract_version_citation()
        assert citation_ver == pyproject_ver, (
            f"CITATION.cff says {citation_ver}, pyproject.toml says {pyproject_ver}"
        )


class TestReadmeTestCount:
    """README's stated test count must be within 5 of actual."""

    def test_readme_test_count_current(self):
        text = README.read_text(encoding="utf-8")
        match = re.search(r"(\d+)\s+tests\s+across", text)
        assert match, "README does not contain a test count (expected 'N tests across')"
        stated = int(match.group(1))

        # Collect actual test count
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        # Last meaningful line: "N tests collected in X.XXs"
        for line in result.stdout.strip().split("\n"):
            m = re.search(r"(\d+)\s+tests?\s+collected", line)
            if m:
                actual = int(m.group(1))
                break
        else:
            pytest.fail("Could not determine actual test count from pytest --collect-only")

        assert abs(stated - actual) <= 5, (
            f"README says {stated} tests but pytest collected {actual}. "
            f"Update the test count in README.md."
        )


class TestReadmeModuleTable:
    """Every public function in __all__ should appear somewhere in README."""

    def test_all_exports_mentioned(self):
        text = README.read_text(encoding="utf-8")
        missing = []
        # Check that major features are mentioned (not every function,
        # but every module category)
        for name in ["descriptives", "moderation", "mediation", "conditional",
                     "cfa", "reliability", "effect", "export", "Word", "CSV"]:
            if name.lower() not in text.lower():
                missing.append(name)
        assert not missing, f"README does not mention: {missing}"
