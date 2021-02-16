from setuptools import setup

setup(
    name="grakn-pytorch-geometric",
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "grakn_pytorch_geometric/_version.py",
    },
    setup_requires=['setuptools_scm'],
    packages=[
        "grakn_pytorch_geometric",
        "grakn_pytorch_geometric.data",
        "grakn_pytorch_geometric.models",
        "grakn_pytorch_geometric.utils",
    ],
    url="",
    license="",
    author="Joren Retel",
    author_email="joren.retel@bayer.com",
    description="",
)
