import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

def plot_column_vs_hours(data, column, y_label, figsize=(10, 4), hour_start=0, ax=None, title=None, show=True):
    """
    Plot a specified column from a dictionary-like or DataFrame against hours (row index).
    - data: dict of lists or pandas.DataFrame
    - column: column name to plot
    - hour_start: 0 or 1 to choose x-axis start (default 0 -> 0..n-1)
    - y_label: label for the y-axis (defaults to column name)
    - returns: matplotlib.figure.Figure
    """
    # ensure DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in data. Available: {list(df.columns)}")

    n = len(df)
    if n == 0:
        raise ValueError("Input data is empty.")

    # build hours axis
    if hour_start == 1:
        hours = list(range(1, n + 1))
    else:
        hours = list(range(0, n))

    # ensure numeric y values
    y = pd.to_numeric(df[column], errors="coerce")
    if y.isna().all():
        raise ValueError(f"Column '{column}' contains no numeric values.")

    sns.set_style("whitegrid")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(hours, y, marker="o", linestyle="-")
    ax.set_xlabel("Hour [h]")
    ax.set_ylabel(y_label or column)
    ax.set_title(title or f"{column} vs Hour")
    ax.set_xlim(min(hours), max(hours))

    if show:
        plt.show()

 
