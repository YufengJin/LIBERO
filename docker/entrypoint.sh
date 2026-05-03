#!/bin/bash
set -e

# Sync sentinel for setup.sh (IsaacGym workflow). Harmless when no setup.sh
# is reading it — just an empty file in /tmp that gets touched at end.
rm -f /tmp/entrypoint_done

export PATH="/opt/venv/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"

# ── 1. Editable install (project mounted at /workspace/libero) ─────
# Both branches resolve install_requires by default. If you need --no-deps
# (e.g. to avoid uv re-resolving heavy science stack), add a `post_install_hooks`
# entry to install_plan.json that re-runs the install with --no-deps.
if [ -f "/workspace/libero/pyproject.toml" ]; then
    echo ">> Installing editable package (pyproject.toml)..."
    cd /workspace/libero && uv pip install -e . --index-strategy unsafe-best-match && cd - > /dev/null
elif [ -f "/workspace/libero/setup.py" ]; then
    echo ">> Installing editable package (setup.py)..."
    cd /workspace/libero && uv pip install -e . --index-strategy unsafe-best-match && cd - > /dev/null
fi

# ── 2. Post-install hooks from InstallationPlan ──────────────────────────────
# Rendered by render_base.py from <repo>/.nautilus/install_plan.json's
# `post_install_hooks`. `when=first_run` entries are wrapped in a sentinel
# guard; `when=every_run` entries fire on every container start.
if [ ! -f /tmp/.nautilus.first-run ]; then
    mkdir -p /workspace/.libero && python -c "
import yaml, os, importlib.util
# Use find_spec so libero's __init__.py is NOT executed (it triggers an
# interactive 'specify dataset path?' prompt when config.yaml is absent).
_spec = importlib.util.find_spec('libero.libero')
_pkg = os.path.dirname(_spec.origin)
_cfg = {
    'benchmark_root': _pkg,
    'bddl_files': os.path.join(_pkg, 'bddl_files'),
    'init_states': os.path.join(_pkg, 'init_files'),
    'datasets': '/workspace/libero/datasets',
    'assets': os.path.join(_pkg, 'assets'),
}
open('/workspace/.libero/config.yaml','w').write(yaml.dump(_cfg))
"
    touch /tmp/.nautilus.first-run
fi

# 
# Slot for downstream sub-skills to inject project-specific steps.

# <<<EXTENSION_ENTRYPOINT_INSERT_ABOVE>>> — sub-skills insert pre-exec hooks above this line

echo ">> Ready."
touch /tmp/entrypoint_done
exec "$@"
