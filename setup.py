from setuptools import setup

setup(
    name="typedb-pytorch-geometric",
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "typedb_pytorch_geometric/_version.py",
    },
    setup_requires=["setuptools_scm"],
    packages=[
        "typedb_pytorch_geometric",
        "typedb_pytorch_geometric.data",
        "typedb_pytorch_geometric.models",
        "typedb_pytorch_geometric.utils",
    ],
    url="",
    license="",
    author="Joren Retel",
    author_email="joren.retel@bayer.com",
    description="",
)
