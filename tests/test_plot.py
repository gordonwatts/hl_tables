from hl_tables import histogram
from dataframe_expressions import DataFrame


def test_histogram():
    df = DataFrame()
    histogram(df, bins=50, range=(0, 20))
