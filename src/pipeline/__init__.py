# src/pipeline_ml/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pipeline")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Exporta utilidades del paquete
from .pipeline import cli as train_cli  # permite: `from pipeline_ml import train_cli`

__all__ = ["__version__", "train_cli"]
