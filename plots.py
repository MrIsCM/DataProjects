import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_series(df, title=None, xlabel=None, ylabel=None, figsize=(12, 6), **kwargs):
	"""
	Plot a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	title: str
		Title of the plot

	xlabel: str
		Label of the x-axis

	ylabel: str
		Label of the y-axis

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)
	ax = df.plot(ax=ax, **kwargs)

	if title is not None:
		ax.set_title(title)

	if xlabel is not None:
		ax.set_xlabel(xlabel)

	if ylabel is not None:
		ax.set_ylabel(ylabel)

	return ax

def plot_series_components(df, title=None, xlabel=None, ylabel=None, figsize=(12, 6), **kwargs):
	"""
	Plot the components of a time series DataFrame

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame with the components to be plotted

	title: str
		Title of the plot

	xlabel: str
		Label of the x-axis

	ylabel: str
		Label of the y-axis

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(len(df.columns), 1, figsize=figsize, sharex=True)

	for i, col in enumerate(df.columns):
		df[col].plot(ax=ax[i], title=col, **kwargs)

	if title is not None:
		ax[0].set_title(title)

	if xlabel is not None:
		ax[-1].set_xlabel(xlabel)

	if ylabel is not None:
		ax[0].set_ylabel(ylabel)

	return ax

def plot_series_heatmap(df, title=None, xlabel=None, ylabel=None, figsize=(12, 6), **kwargs):
	"""
	Plot a heatmap of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	title: str
		Title of the plot

	xlabel: str
		Label of the x-axis

	ylabel: str
		Label of the y-axis

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the heatmap function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(df, ax=ax, **kwargs)

	if title is not None:
		ax.set_title(title)

	if xlabel is not None:
		ax.set_xlabel(xlabel)

	if ylabel is not None:
		ax.set_ylabel(ylabel)

	return ax

def plot_series_pairplot(df, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot a pairplot of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the pairplot function

	Returns:
	--------
	seaborn.axisgrid.PairGrid
		Pairplot of the DataFrame
	
	"""

	fig = sns.pairplot(df, **kwargs)

	if title is not None:
		fig.figure.suptitle(title)

	return fig

def plot_series_jointplot(df, x, y, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot a jointplot of two columns of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	x: str
		Name of the column to be plotted on the x-axis

	y: str
		Name of the column to be plotted on the y-axis

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the jointplot function

	Returns:
	--------
	seaborn.axisgrid.JointGrid
		Jointplot of the DataFrame
	
	"""

	fig = sns.jointplot(x=x, y=y, data=df, **kwargs)

	if title is not None:
		fig.figure.suptitle(title)

	return fig

def plot_series_lag(df, lags=10, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot the lag plot of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	lags: int
		Number of lags to be plotted

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the lag plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Lag plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)

	for i in range(1, lags + 1):
		df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
		sns.scatterplot(x=df.iloc[:, 0], y=df[f'lag_{i}'], ax=ax, label=f'lag_{i}', **kwargs)

	if title is not None:
		ax.set_title(title)

	return ax	

def plot_series_autocorrelation(df, lags=10, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot the autocorrelation plot of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	lags: int
		Number of lags to be plotted

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the autocorrelation plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Autocorrelation plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)
	sns.lineplot(x=np.arange(lags + 1), y=df.iloc[:, 0].autocorr(np.arange(lags + 1)), ax=ax, **kwargs)

	if title is not None:
		ax.set_title(title)

	return ax

def plot_series_partial_autocorrelation(df, lags=10, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot the partial autocorrelation plot of a DataFrame with a DateTimeIndex

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame to be plotted

	lags: int
		Number of lags to be plotted

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the partial autocorrelation plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Partial autocorrelation plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)
	sns.lineplot(x=np.arange(lags + 1), y=df.iloc[:, 0].pacf(np.arange(lags + 1)), ax=ax, **kwargs)

	if title is not None:
		ax.set_title(title)

	return ax

def plot_series_decomposition(df, title=None, figsize=(12, 6), **kwargs):
	"""
	Plot the decomposition of a time series DataFrame

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame with the components to be plotted

	title: str
		Title of the plot

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True)

	df['observed'].plot(ax=ax[0], title='Observed', **kwargs)
	df['trend'].plot(ax=ax[1], title='Trend', **kwargs)
	df['seasonal'].plot(ax=ax[2], title='Seasonal', **kwargs)
	df['residual'].plot(ax=ax[3], title='Residual', **kwargs)

	if title is not None:
		ax[0].set_title(title)

	return ax

def plot_series_forecast(df, forecast, title=None, xlabel=None, ylabel=None, figsize=(12, 6), **kwargs):
	"""
	Plot the forecast of a time series DataFrame

	Parameters:
	-----------
	df: pandas.DataFrame
		DataFrame with the observed values

	forecast: pandas.DataFrame
		DataFrame with the forecasted values

	title: str
		Title of the plot

	xlabel: str
		Label of the x-axis

	ylabel: str
		Label of the y-axis

	figsize: tuple
		Size of the figure

	kwargs: dict
		Additional arguments to be passed to the plot function

	Returns:
	--------
	matplotlib.axes._subplots.AxesSubplot
		Plot of the DataFrame
	
	"""

	fig, ax = plt.subplots(figsize=figsize)
	df.plot(ax=ax, **kwargs)
	forecast.plot(ax=ax, **kwargs)

	if title is not None:
		ax.set_title(title)

	if xlabel is not None:
		ax.set_xlabel(xlabel)

	if ylabel is not None:
		ax.set_ylabel(ylabel)

	return ax