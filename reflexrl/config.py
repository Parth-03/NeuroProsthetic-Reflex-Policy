"""Shared command-line options and run configs."""

import json
import os

from reflexrl.env import DEFAULT_WRAPPER_KWARGS, ENV_ID

CONFIG_NAME = "config.json"


def add_wrapper_args(parser, defaults=None):
    """Adds one option per wrapper setting. Unset options stay None."""
    defaults = defaults or DEFAULT_WRAPPER_KWARGS
    g = parser.add_argument_group("environment wrapper")
    g.add_argument("--env-id", default=None, help=f"MyoSuite env id (default {ENV_ID})")
    g.add_argument("--residual-scale", type=float, default=None,
                   help=f"max |residual| per muscle (default {defaults['residual_scale']})")
    g.add_argument("--dead-zone", type=float, default=None,
                   help=f"reflex dead zone, rad (default {defaults['dead_zone']})")
    g.add_argument("--reflex-gain", type=float, default=None,
                   help=f"reflex gain k_p (default {defaults['reflex_gain']})")
    g.add_argument("--pain-threshold", type=float, default=None,
                   help=f"activation that triggers pain (default {defaults['pain_threshold']})")
    g.add_argument("--withdraw-steps", type=int, default=None,
                   help="fixed withdrawal duration in steps (default: while pain lasts)")
    g.add_argument("--safe-offset", type=float, default=None,
                   help=f"safe target offset, rad (default {defaults['safe_offset']})")
    g.add_argument("--lambda-pain", type=float, default=None,
                   help=f"pain penalty weight (default {defaults['lambda_pain']})")
    g.add_argument("--random-pain-prob", type=float, default=None,
                   help=f"per-step random pain probability (default {defaults['random_pain_prob']})")
    g.add_argument("--random-pain-steps", type=int, default=None,
                   help=f"random pain event length (default {defaults['random_pain_steps']})")
    return parser


def wrapper_kwargs_from_args(args, base=None):
    """Combines ``base`` settings with any options given on the command line."""
    kwargs = dict(DEFAULT_WRAPPER_KWARGS)
    kwargs.update(base or {})
    for key in DEFAULT_WRAPPER_KWARGS:
        value = getattr(args, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def save_config(run_dir, config):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CONFIG_NAME)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def load_config(run_dir):
    path = os.path.join(run_dir, CONFIG_NAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_model_path(path):
    """Accepts a run folder or a model file and returns (run_dir, model_file)."""
    if os.path.isdir(path):
        return path, os.path.join(path, "model.zip")
    return os.path.dirname(path) or ".", path
