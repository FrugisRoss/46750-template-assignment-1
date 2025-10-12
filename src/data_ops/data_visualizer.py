import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import warnings

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



def plot_columns_vs_hours(data,
                          columns,
                          labels=None,
                          y_label=None,
                          figsize=(10, 4),
                          hour_start=0,
                          ax=None,
                          title=None,
                          show=True,
                          max_series=5):
    """
    Plot multiple specified columns from a dictionary-like or DataFrame against hours (row index).
    - data: dict of lists or pandas.DataFrame
    - columns: list/tuple of column names to plot (len <= max_series)
    - labels: None, dict mapping column -> legend label, or list of labels (same order as columns)
    - y_label: label for the y-axis (defaults to None)
    - hour_start: 0 or 1 to choose x-axis start (default 0 -> 0..n-1)
    - ax: matplotlib Axes to plot on (optional)
    - title: plot title (optional)
    - show: whether to call plt.show()
    - returns: matplotlib.figure.Figure
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    if not isinstance(columns, (list, tuple)):
        raise TypeError("`columns` must be a list or tuple of column names.")

    if len(columns) == 0:
        raise ValueError("`columns` must contain at least one column name.")

    if len(columns) > max_series:
        raise ValueError(f"Too many series to plot (got {len(columns)}). Max allowed is {max_series}.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}. Available: {list(df.columns)}")

    n = len(df)
    if n == 0:
        raise ValueError("Input data is empty.")

    # build hours axis
    if hour_start == 1:
        hours = list(range(1, n + 1))
    else:
        hours = list(range(0, n))

    # prepare labels
    if labels is None:
        legend_labels = list(columns)
    elif isinstance(labels, dict):
        legend_labels = [labels.get(c, c) for c in columns]
    elif isinstance(labels, (list, tuple)):
        if len(labels) != len(columns):
            raise ValueError("If `labels` is a list/tuple it must have the same length as `columns`.")
        legend_labels = list(labels)
    else:
        raise TypeError("`labels` must be None, a dict, or a list/tuple.")

    # plotting setup
    markers = ['o', 'D', '^', 's', 'v']  # up to 5
    linestyles = ['-', '--', '-.', ':', '-']
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plotted_any = False
    for i, col in enumerate(columns):
        y = pd.to_numeric(df[col], errors="coerce")
        if y.isna().all():
            warnings.warn(f"Column '{col}' contains no numeric values and will be skipped.")
            continue

        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        ax.plot(hours, y, marker=m, linestyle=ls, label=legend_labels[i])
        plotted_any = True

    if not plotted_any:
        raise ValueError("No numeric series were plotted; all requested columns contain non-numeric values.")

    ax.set_xlabel("Hour [h]")
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_title(title or ", ".join(legend_labels))
    ax.set_xlim(min(hours), max(hours))
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig


def plot_sensitivity_vs_hours(solutions_dict, column, y_label, figsize=(10, 4), hour_start=0,
                              title=None, ax=None, show=True, legend_title=None):
    """
    Plot a given column (e.g., 'd' or 'z') vs hours for multiple penalty values.

    Args:
        solutions_dict: dict {penalty_value: solution_dict}
        column: string, column name to plot
        y_label: string, y-axis label
        legend_title: optional string for the legend title (default: None)
        ax: matplotlib axis to plot on (optional)
        show: whether to call plt.show() (default: True)
    """


    sns.set_style("whitegrid")

    # Use provided axis, or create a new figure/axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # get the figure from the provided axis

    for penalty, sol in solutions_dict.items():
        df = pd.DataFrame(sol)
        if column not in df.columns:
            continue  # skip if not in solution

        n = len(df)
        hours = list(range(hour_start, hour_start + n))
        y = pd.to_numeric(df[column], errors="coerce")

        ax.plot(hours, y, marker="o", linestyle="-", label=f"{penalty}")

    ax.set_xlabel("Hour [h]")
    ax.set_ylabel(y_label)
    ax.set_title(f"{column} vs Hour" if not title else title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax
