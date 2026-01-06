from setuptools import setup, find_packages
from pathlib import Path


def _parse_requirements(file_path):
    requirements = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line.split(";")[0].strip())
    return requirements


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    requirements_path = base_dir / "requirements.txt"
    
    setup(
        name="diffusers_lite",
        version="0.0.1",
        packages=find_packages(),
        install_requires=_parse_requirements(requirements_path),
        author="",
    )