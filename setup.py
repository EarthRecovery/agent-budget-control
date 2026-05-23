import os
from setuptools import find_packages, setup


def _read_requirements(relative_path):
    requirements_path = os.path.join(os.path.dirname(__file__), relative_path)
    if not os.path.exists(requirements_path):
        return []
    with open(requirements_path) as requirements_file:
        return [
            line.strip()
            for line in requirements_file
            if line.strip() and not line.startswith("#")
        ]

# Base dependencies required for all installations
base_requires = [
    "IPython",
    "matplotlib",
    "gym",
    "gym_sokoban",
    "peft",
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    # "flash-attn==2.7.4.post1",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "pybind11",
    "ray[default]>=2.41.0",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers",
    "vllm>=0.8.5,<=0.11.0",
    "wandb",
    "gymnasium",
    "gymnasium[toy-text]",
    "pyarrow>=19.0.0",
    "pylatexenc",
    "torchdata",
    "debugpy",
    "together",
    "anthropic",
    "faiss-cpu==1.11.0",
]

# Optional dependencies for webshop environment
webshop_requires = [
    "beautifulsoup4",
    "cleantext",
    "flask",
    "html2text",
    "rank_bm25",
    "pyserini",
    "thefuzz",
    "gdown",
    "spacy",
    "rich",
]

# Optional dependencies for lean environment
lean_requires = [
    "kimina-client",
]

# Optional dependencies for search environment
# Note: the retrieval server (scripts/retrieval/server.py) additionally needs:
#   flask, faiss-cpu, sentence-transformers
search_requires = [
    "requests",
]

# Optional dependencies for Robotouille environment
robotouille_requires = _read_requirements("external/robotouille/requirements.txt")

setup(
    name='ragen',
    version='0.1',
    package_dir={'': '.'},
    packages=find_packages(include=['ragen', 'ragen.*']),
    author='',
    author_email='',
    acknowledgements='',
    description='',
    install_requires=base_requires,
    extras_require={
        "webshop": webshop_requires,
        "lean": lean_requires,
        "search": search_requires,
        "robotouille": robotouille_requires,
        "all": webshop_requires + lean_requires + search_requires + robotouille_requires,
    },
    package_data={'ragen': ['*/*.md']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)
