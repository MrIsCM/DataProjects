import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_freq(df, freq=None):
	"""
	Set frequency of DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame with DateTimeIndex

	freq: str
		Frequency to be set
		if None, the frequency is inferred from the DataFrame index

	Returns:
	--------
	pandas.DataFrame
		DataFrame with the new frequency
	
	"""

	if freq is None:
		freq = pd.infer_freq(df.index)

	return df.asfreq(freq)

def extend_time(df, t_min=None, t_max=None, dt=None):
	"""
	Extend the time index of the dataframe 'df' from 't_min' to 't_max' and a 'dt' interval.
	Some parameters might be omited.

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be extended

	t_min: pd.Timestamp
		Minimum time index
		if None, the minimum time index of the DataFrame is used

	t_max: pd.Timestamp
		Maximum time index
		if None, the maximum time index of the DataFrame is used

	dt: int or pd.Timedelta
		Time interval to be added to the time index
		
		- if int:
			- if positive, dt periods are added to t_max
			- if negative, dt periods are subtracted from t_min
		
		- else, it is converted to pd.Timedelta
			- if positive, dt is added to t_max
			- if negative, dt is subtracted from t_min

	If all the parameters are None, dt = 1.

	Returns:
	--------
	pandas.DataFrame
		DataFrame with extended time index
	
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

	return df.reindex(new_index)


def RMSE(y_true, y_pred):
	"""
	Calculate the Root Mean Squared Error (RMSE) between two arrays

	Parameters:
	-----------
	y_true: array-like
		True values

	y_pred: array-like
		Predicted values

	Returns:
	--------
	float
		RMSE value

	"""

	return np.sqrt(np.mean((y_true - y_pred)**2))

def MAE(y_true, y_pred):
	"""
	Calculate the Mean Absolute Error (MAE) between two arrays

	Parameters:
	-----------
	y_true: array-like
		True values

	y_pred: array-like
		Predicted values

	Returns:
	--------
	float
		MAE value

	"""

	return np.mean(np.abs(y_true - y_pred))

def MAPE(y_true, y_pred):
	"""
	Calculate the Mean Absolute Percentage Error (MAPE) between two arrays

	Parameters:
	-----------
	y_true: array-like
		True values

	y_pred: array-like
		Predicted values

	Returns:
	--------
	float
		MAPE value

	"""

	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Test function
def rlabel(ax, label=None, **kw):
    """Add a right-side axis title."""
    if label is None:
        ax, label = plt.gca(), ax
    bbox = kw.pop('bbox', dict(facecolor='.9', alpha=0.2))
    ax.text(1, .5, label,
            rotation=-90, ha='left', va='center', transform=ax.transAxes,
            bbox=bbox, **kw)