#!/usr/bin/env bash
# Entrypoint for the training container.
#
# Makes the project's Python modules importable, then execs the command.
set -e

export PYTHONPATH="/workspace:${PYTHONPATH}"
cd /workspace
exec "$@"