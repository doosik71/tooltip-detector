"""One place for the progress-bar settings the runnable scripts share.

Plain ASCII and a fixed width, on purpose:

  ascii=True   the Unicode block characters tqdm prefers render as mojibake in
               a redirected log, in a Windows console and over some SSH
               terminals; `#` renders everywhere.
  ncols        without a fixed width tqdm re-measures the terminal on every
               refresh, so the bar reflows when the window is resized and
               collapses to a useless width when stdout is not a terminal at
               all (training is usually launched with the output redirected to
               a log file). A fixed width keeps every line the same length.
"""

from tqdm import tqdm as _tqdm

BAR_WIDTH = 100
BAR_ASCII = True


def progress(iterable=None, **kwargs):
    """`tqdm` with this project's defaults; any keyword can still override them."""
    kwargs.setdefault("ascii", BAR_ASCII)
    kwargs.setdefault("ncols", BAR_WIDTH)
    return _tqdm(iterable, **kwargs)
