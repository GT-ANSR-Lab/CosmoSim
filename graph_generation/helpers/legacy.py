"""Legacy wrappers for backward compatibility with constellation generation scripts."""
from graph_generation.helpers.generate_basic_graphs import generate_basic_graphs


def generate_all_graphs(*args, **kwargs):
    """Alias for generate_basic_graphs to maintain compatibility."""
    return generate_basic_graphs(*args, **kwargs)


def generate_all_graphs_shells_failure(*_args, **_kwargs):
    raise NotImplementedError(
        "Shell-failure graph generation is not implemented in the public release."
    )
