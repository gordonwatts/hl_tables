from dataframe_expressions import DataFrame
from hl_tables import histogram
import pytest


@pytest.fixture
def call_make_local_hist(mocker):
    'Return a dummy histogram from a make local call'

    def rtn_hist():
        bins = (0.0, 1.0, 2.0)
        data = (20, 30)
        return data, bins

    return mocker.patch('hl_tables.local.make_local', side_effect=lambda df: rtn_hist())


@pytest.fixture
def mock_plotting(mocker):
    mocker.patch('matplotlib.pyplot.subplots',
                 return_value=(mocker.MagicMock(), mocker.MagicMock()))
    mocker.patch('mplhep.histplot')


def test_histogram(call_make_local_hist, mock_plotting):
    df = DataFrame()
    histogram(df, bins=50, range=(0, 20))

    call_make_local_hist.assert_called_once_with(df)
