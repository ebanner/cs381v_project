# Defines an elapsed time class that automatically formats time to seconds,
# minutes, or hours, depending on the time taken.

import time


class ElapsedTimer(object):
  """A class that makes printing elapsed time easier."""

  def __init__(self):
    """Starts a timer at the moment of construction."""
    self._start_time = time.time()

  def reset(self):
    """Restarts the timer by setting the start time to now."""
    self._start_time = time.time()

  def get_elapsed_time(self):
    """Returns the elapsed time, formatted as a string.
    
    Args:
      start_time: the start time (called before timing using time.time()).

    Returns:
      The elapsed time as a string (e.g. "x seconds" or "x minutes").
    """
    elapsed = time.time() - self._start_time
    time_units = ['seconds', 'minutes', 'hours', 'days']
    unit_index = 0
    intervals = [60, 60, 24]
    for interval in intervals:
      if elapsed < interval:
        break
      elapsed /= interval
      unit_index += 1
    elapsed = round(elapsed, 2)
    return '{} {}'.format(elapsed, time_units[unit_index])
