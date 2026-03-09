"""Backwards-compatible re-export — moved to langloc.eval.sort_table."""
import warnings as _warnings

_warnings.warn(
    "langloc.localization.sort_eval_table is deprecated; "
    "use langloc.eval.sort_table instead.",
    DeprecationWarning,
    stacklevel=2,
)

from langloc.eval.sort_table import *  # noqa: F401,F403

if __name__ == "__main__":
    from langloc.eval.sort_table import main
    main()
