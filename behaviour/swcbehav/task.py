"""The visual change-detection task (open-loop, idealized).

This module builds the *stimulus stream* a mouse experiences: a sequence of
image flashes, some of which are "changes" (a new image) and some of which are
"omissions" (the image is withheld and the screen stays gray). It contains no
behavior -- see ``generate.py`` for the mouse.

Idealizations relative to Piet et al. (2024), all deliberate for teaching:

* **Open-loop.** In the real task, premature licking delays the next change.
  Here the change schedule is fixed in advance and does not depend on licking.
  This removes a feedback loop that complicates data generation without adding
  anything to the *statistics* the students are learning.
* Each flash is 750 ms (250 ms image + 500 ms gray), as in the paper.
* Change times are drawn from a geometric distribution, as in the paper.
* 5% of flashes are omitted, except change flashes and the flash immediately
  before a change (which the paper also never omits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLASH_DURATION = 0.75  # seconds per image presentation (250 ms image + 500 ms gray)


def make_session(
    n_images: int = 4800,
    n_image_ids: int = 8,
    change_p: float = 0.25,
    min_run: int = 2,
    omission_prob: float = 0.05,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Build one session's stimulus stream.

    Parameters
    ----------
    n_images : number of image flashes in the session (paper: ~4800 == 1 hour).
    n_image_ids : size of the image set (paper: 8 familiar images).
    change_p : per-flash change hazard. Run lengths are ``min_run + Geometric(change_p)``,
        so the mean run length is ``min_run + 1 / change_p``.
    min_run : minimum number of repeats before a change can occur.
    omission_prob : probability an eligible flash is omitted.
    seed : RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame with one row per flash and columns:
        image_index, time, image_id, is_change, is_omission, is_post_omission.
    """
    rng = np.random.default_rng(seed)

    image_ids: list[int] = []
    is_change: list[bool] = []
    current_id = int(rng.integers(n_image_ids))
    block = 0
    while len(image_ids) < n_images:
        run_len = min_run + int(rng.geometric(change_p))
        for j in range(run_len):
            if len(image_ids) >= n_images:
                break
            image_ids.append(current_id)
            is_change.append(block > 0 and j == 0)
        block += 1
        new_id = current_id
        while new_id == current_id:
            new_id = int(rng.integers(n_image_ids))
        current_id = new_id

    image_ids = np.asarray(image_ids)
    is_change = np.asarray(is_change, dtype=bool)
    n = len(image_ids)

    # Omissions: not on changes, and not on the flash right before a change.
    pre_change = np.zeros(n, dtype=bool)
    pre_change[:-1] = is_change[1:]
    eligible = (~is_change) & (~pre_change)
    is_omission = eligible & (rng.random(n) < omission_prob)

    is_post_omission = np.zeros(n, dtype=bool)
    is_post_omission[1:] = is_omission[:-1]

    return pd.DataFrame(
        {
            "image_index": np.arange(n),
            "time": np.arange(n) * FLASH_DURATION,
            "image_id": image_ids,
            "is_change": is_change,
            "is_omission": is_omission,
            "is_post_omission": is_post_omission,
        }
    )