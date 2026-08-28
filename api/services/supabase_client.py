"""Compatibility import for the domain repository during route extraction."""
import sys
from repositories import transport as _repository
sys.modules[__name__] = _repository
