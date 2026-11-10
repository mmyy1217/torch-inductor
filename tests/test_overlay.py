"""Verify that overlays use an explicit checkout and preserve the installed package."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_overlay", ROOT / "prepare_python_overlay.py"
)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


class OverlayTests(unittest.TestCase):
    def test_explicit_source_preserves_wheel_and_replaces_inductor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "source"
            implementation = repo / "torch_npu/_inductor/kernel"
            implementation.mkdir(parents=True)
            (implementation / "tilelang_flex_attention.py").write_text(
                "VALUE = 'new'\n"
            )
            scheduling = (
                implementation.parent / "tilelang/codegen/tilelang_scheduling.py"
            )
            scheduling.parent.mkdir(parents=True)
            scheduling.write_text("# Integrated codegen fixture\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Overlay Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "Test fixture",
                ],
                cwd=repo,
                check=True,
            )
            package = root / "wheel/torch_npu"
            (package / "_inductor").mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "_C.so").write_bytes(b"installed-extension")
            (package / "_inductor/obsolete.py").write_text("OLD = True\n")
            fake = {
                "torch": SimpleNamespace(__version__="2.14.0"),
                "torch_npu": SimpleNamespace(
                    __version__="2.14.0", __file__=str(package / "__init__.py")
                ),
            }
            dest = root / "overlay"
            old_path = sys.path[:]
            try:
                with patch.dict(sys.modules, fake):
                    overlay.main(["--source-repo", str(repo), "--output", str(dest)])
                    for forbidden in (dest, repo / "nested", package / "nested"):
                        with self.assertRaises(SystemExit):
                            overlay.main(
                                ["--source-repo", str(repo), "--output", str(forbidden)]
                            )
            finally:
                sys.path[:] = old_path
            self.assertTrue((package / "_inductor/obsolete.py").is_file())
            self.assertFalse((dest / "torch_npu/_inductor/obsolete.py").exists())
            self.assertEqual(
                (dest / "torch_npu/_C.so").read_bytes(), b"installed-extension"
            )
            self.assertEqual(
                (
                    dest / "torch_npu/_inductor/kernel/tilelang_flex_attention.py"
                ).read_text(),
                "VALUE = 'new'\n",
            )
            manifest = json.loads((dest / "overlay.json").read_text())
            self.assertEqual(manifest["source_repo"], str(repo))
            self.assertEqual(len(manifest["source_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
