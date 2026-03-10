"""
Unit tests targeting the Unknown dtype warning path in pyfunctions.py (lines 67-70).

This covers the branch:
    elif not df_copy.implementation.is_pandas() and df_copy[var_name].dtype==nw.Unknown:
        msg = ...
        warnings.warn(msg, RuntimeWarning)
        continue
"""

import unittest
import warnings
from unittest.mock import MagicMock, patch, PropertyMock

import polars as pl
import narwhals.stable.v2 as nw

from pyreadstat.pyfunctions import set_value_labels


class MockMetadata:
    """Mock metadata object that mimics pyreadstat metadata_container."""

    def __init__(self, value_labels=None, variable_to_label=None,
                 variable_value_labels=None):
        self.value_labels = value_labels
        self.variable_to_label = variable_to_label
        self.variable_value_labels = variable_value_labels or {}


class TestUnknownDtypeWarning(unittest.TestCase):
    """Tests for the Unknown dtype warning path in set_value_labels (lines 67-70)."""

    def test_unknown_dtype_emits_warning_and_skips(self):
        """When a polars column has Unknown dtype, a RuntimeWarning should be emitted
        and the column should be skipped (not transformed)."""
        # Create a polars DataFrame with a normal column
        df = pl.DataFrame({"status": [1.0, 2.0, 3.0]})
        meta = MockMetadata(
            value_labels={"status_fmt": {1.0: "Active", 2.0: "Inactive", 3.0: "Pending"}},
            variable_to_label={"status": "status_fmt"},
        )

        # We need to mock the narwhals wrapper so that the dtype check returns nw.Unknown
        # The flow is: nw.from_native(df).clone() -> df_copy, then df_copy[var_name].dtype is checked
        # We'll patch at the narwhals level to simulate Unknown dtype
        original_set_value_labels = set_value_labels

        # A more targeted approach: patch the narwhals Series dtype property
        # after the clone but before the dtype check
        with patch("narwhals.stable.v2.from_native") as mock_from_native:
            # Create a real narwhals frame first
            real_nw_frame = nw.from_native(df).clone()

            # Create a mock series that returns Unknown dtype
            mock_series = MagicMock()
            mock_series.dtype = nw.Unknown
            mock_series.to_list.return_value = [1.0, 2.0, 3.0]
            mock_series.__len__ = lambda self: 3
            mock_series.null_count.return_value = 0

            # Create a mock frame
            mock_frame = MagicMock()
            mock_frame.clone.return_value = mock_frame
            mock_frame.implementation.is_pandas.return_value = False
            mock_frame.columns = ["status"]
            mock_frame.__getitem__ = lambda self, key: mock_series
            mock_frame.to_native.return_value = df

            mock_from_native.return_value = mock_frame

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = set_value_labels(df, meta, formats_as_category=False)
                runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
                self.assertTrue(
                    len(runtime_warnings) > 0,
                    "Expected RuntimeWarning about unknown/not supported data type"
                )
                self.assertIn("unknown/not supported data type", str(runtime_warnings[0].message))

    def test_unknown_dtype_with_formats_as_category(self):
        """Unknown dtype path should also trigger with formats_as_category=True."""
        df = pl.DataFrame({"code": [1.0, 2.0]})
        meta = MockMetadata(
            value_labels={"code_fmt": {1.0: "A", 2.0: "B"}},
            variable_to_label={"code": "code_fmt"},
        )

        with patch("narwhals.stable.v2.from_native") as mock_from_native:
            mock_series = MagicMock()
            mock_series.dtype = nw.Unknown
            mock_series.to_list.return_value = [1.0, 2.0]
            mock_series.__len__ = lambda self: 2
            mock_series.null_count.return_value = 0

            mock_frame = MagicMock()
            mock_frame.clone.return_value = mock_frame
            mock_frame.implementation.is_pandas.return_value = False
            mock_frame.columns = ["code"]
            mock_frame.__getitem__ = lambda self, key: mock_series
            mock_frame.to_native.return_value = df

            mock_from_native.return_value = mock_frame

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = set_value_labels(df, meta, formats_as_category=True)
                runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
                self.assertTrue(len(runtime_warnings) > 0)
                self.assertIn("unknown/not supported data type", str(runtime_warnings[0].message))


if __name__ == "__main__":
    unittest.main()
