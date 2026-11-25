"""Translate legacy environment names at cluster process boundaries."""
import os


def normalize_cluster_environment(environ=None):
    """Accept old names; an explicitly supplied CLUSTER_ value wins."""
    env = os.environ if environ is None else environ
    for key, value in tuple(env.items()):
        if key.startswith("DUAL_"):
            env.setdefault("CLUSTER_" + key[5:], value)
    return env
