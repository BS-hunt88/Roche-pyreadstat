"""
Extended unit tests for pyreadstat/worker.py.

Tests additional edge cases including:
- Exception propagation from read functions
- Worker with various kwarg combinations
- Worker with None values
- Worker return type verification
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from pyreadstat.worker import worker


class TestWorkerExceptionHandling(unittest.TestCase):
    """Tests for worker() exception handling behavior."""

    def test_worker_propagates_file_not_found(self):
        """Worker should propagate FileNotFoundError from the read function."""
        def failing_read(path, row_offset=0, row_limit=0):
            raise FileNotFoundError(f"File not found: {path}")

        with self.assertRaises(FileNotFoundError):
            worker((failing_read, "/fake/nonexistent.sav", 0, 100, {}))

    def test_worker_propagates_value_error(self):
        """Worker should propagate ValueError from the read function."""
        def failing_read(path, row_offset=0, row_limit=0):
            raise ValueError("Invalid value")

        with self.assertRaises(ValueError):
            worker((failing_read, "/fake/path.sav", 0, 100, {}))

    def test_worker_propagates_runtime_error(self):
        """Worker should propagate RuntimeError from the read function."""
        def failing_read(path, row_offset=0, row_limit=0):
            raise RuntimeError("Runtime failure")

        with self.assertRaises(RuntimeError):
            worker((failing_read, "/fake/path.sav", 0, 100, {}))

    def test_worker_propagates_type_error(self):
        """Worker should propagate TypeError from the read function."""
        def failing_read(path, row_offset=0, row_limit=0):
            raise TypeError("Type error in read")

        with self.assertRaises(TypeError):
            worker((failing_read, "/fake/path.sav", 0, 100, {}))


class TestWorkerReturnTypes(unittest.TestCase):
    """Tests verifying worker return types with various DataFrame types."""

    def test_worker_returns_dataframe_with_numeric_data(self):
        """Worker should return DataFrame with numeric columns."""
        mock_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 10, {}))
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["a", "b"])

    def test_worker_returns_dataframe_with_string_data(self):
        """Worker should return DataFrame with string columns."""
        mock_df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 10, {}))
        self.assertEqual(result["name"].tolist(), ["Alice", "Bob"])

    def test_worker_returns_dataframe_with_nan(self):
        """Worker should return DataFrame with NaN values intact."""
        mock_df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 10, {}))
        self.assertTrue(np.isnan(result["val"].iloc[1]))

    def test_worker_returns_dataframe_with_mixed_types(self):
        """Worker should return DataFrame with mixed column types."""
        mock_df = pd.DataFrame({"num": [1.0], "text": ["hello"], "flag": [True]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 10, {}))
        self.assertEqual(len(result.columns), 3)


class TestWorkerKwargsVariations(unittest.TestCase):
    """Tests for worker() with various kwarg combinations."""

    def test_worker_with_metadataonly_kwarg(self):
        """Worker should pass metadataonly kwarg to read function."""
        mock_df = pd.DataFrame()
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        worker((mock_read_fn, "/path", 0, 10, {"metadataonly": True}))
        mock_read_fn.assert_called_once_with(
            "/path", row_offset=0, row_limit=10, metadataonly=True
        )

    def test_worker_with_encoding_kwarg(self):
        """Worker should pass encoding kwarg to read function."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        worker((mock_read_fn, "/path", 0, 10, {"encoding": "UTF-8"}))
        mock_read_fn.assert_called_once_with(
            "/path", row_offset=0, row_limit=10, encoding="UTF-8"
        )

    def test_worker_with_multiple_kwargs(self):
        """Worker should pass multiple kwargs correctly."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        kwargs = {
            "usecols": ["x", "y"],
            "disable_datetime_conversion": True,
            "encoding": "LATIN1",
        }
        worker((mock_read_fn, "/path", 5, 20, kwargs))
        mock_read_fn.assert_called_once_with(
            "/path", row_offset=5, row_limit=20,
            usecols=["x", "y"],
            disable_datetime_conversion=True,
            encoding="LATIN1",
        )

    def test_worker_with_output_format_kwarg(self):
        """Worker should pass output_format kwarg."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        worker((mock_read_fn, "/path", 0, 10, {"output_format": "dict"}))
        mock_read_fn.assert_called_once_with(
            "/path", row_offset=0, row_limit=10, output_format="dict"
        )


class TestWorkerBoundaryConditions(unittest.TestCase):
    """Tests for worker() with boundary values."""

    def test_worker_with_zero_row_limit(self):
        """Worker with row_limit=0 (means no limit)."""
        mock_df = pd.DataFrame({"x": [1, 2, 3]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 0, {}))
        mock_read_fn.assert_called_once_with("/path", row_offset=0, row_limit=0)
        self.assertEqual(len(result), 3)

    def test_worker_with_large_offset(self):
        """Worker with very large row_offset."""
        mock_df = pd.DataFrame()
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 999999, 10, {}))
        mock_read_fn.assert_called_once_with("/path", row_offset=999999, row_limit=10)
        self.assertTrue(result.empty)

    def test_worker_with_large_row_limit(self):
        """Worker with very large row_limit."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 999999, {}))
        mock_read_fn.assert_called_once_with("/path", row_offset=0, row_limit=999999)

    def test_worker_with_list_input(self):
        """Worker should also work with list input (not just tuple)."""
        mock_df = pd.DataFrame({"x": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker([mock_read_fn, "/path", 0, 10, {}])
        self.assertTrue(result.equals(mock_df))


if __name__ == "__main__":
    unittest.main()
