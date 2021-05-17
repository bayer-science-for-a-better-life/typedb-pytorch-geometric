__author__ = ["Joren Retel"]
__email__ = ["joren.retel@bayer.com"]


try:
    from typedb_pytorch_geometric._version import version as __version__
except ImportError:
    # this protects against an edge case where a user tries to import
    # the module without installing it, by adding it manually to
    # sys.path or trying to run it directly from the git checkout.
    __version__ = "not-installed"
