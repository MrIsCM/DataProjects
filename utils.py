import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_freq(df, freq=None):
	"""
	Set frequency of DateTimeIndex
	"""
	if freq is None:
		freq = pd.infer_freq(df.index)

	return df.asfreq(freq)

def extend_time(df, t_min=None, t_max=None, dt=None):
	"""
	Extend the time index of the dataframe df from t_min to t_max with and a dt interval.
	Some parameters might be omited
	"""

	freq = df.index.freq
	
	if t_min is t_max is dt is None:
		dt = 1

	if t_min is None:
		t_min = df.index.min()

	if t_max is None:
		t_max = df.index.max()

	if dt is not None:
		if isinstance(dt, int):
			if dt > 0:
				t_max += dt*freq 
			else:
				t_min -= dt*freq
		else:
			dt = pd.to_timedelta(dt)

			if dt > pd.Timedelta(0):
				t_max += dt
			else:
				t_min -= dt

	new_index = pd.date_range(start=t_min, end=t_max, freq=freq)

	return df.reinfex(new_index)