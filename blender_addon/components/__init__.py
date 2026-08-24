"""Component builder registry."""
from .analytic import ANALYTIC_BUILDERS
from .noise import APPROXIMATE_BUILDERS
from .graph import shader_graph

BUILDERS = {**ANALYTIC_BUILDERS, **APPROXIMATE_BUILDERS}
BUILDERS["shader_graph"] = shader_graph
APPROXIMATE_TYPES = frozenset(APPROXIMATE_BUILDERS)
