"""
Unit tests for pyreadstat/worker.py.

Tests the worker() function used for multiprocessing read operations.
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from pyreadstat.worker import worker


class TestWorker(unittest.TestCase):
    """Tests for the worker() function."""

    def test_worker_basic_call(self):
        """Worker should call the read function with correct arguments and return df."""
        mock_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        path = "/fake/path/file.sav"
        row_offset = 0
        row_limit = 100
        kwargs = {}

        result = worker((mock_read_fn, path, row_offset, row_limit, kwargs))

        mock_read_fn.assert_called_once_with(
            path, row_offset=row_offset, row_limit=row_limit
        )
        self.assertTrue(result.equals(mock_df))

    def test_worker_with_kwargs(self):
        """Worker should pass additional kwargs to the read function."""
        mock_df = pd.DataFrame({"b": [4, 5]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        path = "/fake/path/file.sas7bdat"
        row_offset = 10
        row_limit = 50
        kwargs = {"disable_datetime_conversion": True, "usecols": ["b"]}

        result = worker((mock_read_fn, path, row_offset, row_limit, kwargs))

        mock_read_fn.assert_called_once_with(
            path,
            row_offset=row_offset,
            row_limit=row_limit,
            disable_datetime_conversion=True,
            usecols=["b"],
        )
        self.assertTrue(result.equals(mock_df))

    def test_worker_returns_only_dataframe(self):
        """Worker should return only the dataframe, not the metadata."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_meta.number_rows = 1
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 10, {}))

        # Result should be just the dataframe
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.equals(mock_df))

    def test_worker_with_offset_and_limit(self):
        """Worker should pass correct row_offset and row_limit."""
        mock_df = pd.DataFrame({"c": [3, 4]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        worker((mock_read_fn, "/path", 50, 25, {}))

        mock_read_fn.assert_called_once_with(
            "/path", row_offset=50, row_limit=25
        )

    def test_worker_with_empty_dataframe(self):
        """Worker should handle empty dataframe results correctly."""
        mock_df = pd.DataFrame()
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 0, {}))
        self.assertTrue(result.empty)

    def test_worker_input_is_tuple(self):
        """Worker takes a single tuple argument (designed for multiprocessing.Pool.map)."""
        mock_df = pd.DataFrame({"a": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        # Should work with tuple input
        input_tuple = (mock_read_fn, "/path", 0, 10, {})
        result = worker(input_tuple)
        self.assertTrue(result.equals(mock_df))


if __name__ == "__main__":
    unittest.main()
