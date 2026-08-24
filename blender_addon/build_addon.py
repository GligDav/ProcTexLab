"""Build a clean, installable add-on ZIP without caches or development data."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE.parent / "ptk_blender_addon-1.0.2.zip"
EXCLUDED_PARTS = {"__pycache__", "tests"}
EXCLUDED_FILES = {"IMPLEMENTATION_PLAN.md", "example_result.json", "build_addon.py"}


def main() -> None:
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PACKAGE.rglob("*.py")):
            relative = path.relative_to(PACKAGE)
            if any(part in EXCLUDED_PARTS for part in relative.parts) or path.name in EXCLUDED_FILES:
                continue
            archive.write(path, Path(PACKAGE.name) / relative)
    print(OUTPUT)


if __name__ == "__main__":
    main()
