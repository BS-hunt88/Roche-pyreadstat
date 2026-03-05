"""
Extended write/read round-trip tests covering additional formats, data types,
and edge cases not covered by existing tests.

Tests include:
- DTA round-trip with string columns
- XPT v5 round-trip
- POR round-trip with multiple columns
- Round-trips with date columns
- Round-trips with NaN in various positions
- Round-trips with unicode/international characters
- Round-trips with very long strings
- Write with variable_value_labels and read back
- Write with missing_ranges and read back
- Multiple format version writes
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import pyreadstat


class TestDTARoundTripExtended(unittest.TestCase):
    """Extended DTA round-trip tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dta_roundtrip_string_column(self):
        """DTA round-trip with string-only column."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        path = os.path.join(self.tmpdir, "strings.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(df_read["name"].tolist(), ["Alice", "Bob", "Charlie"])

    def test_dta_roundtrip_mixed_columns(self):
        """DTA round-trip with mixed numeric and string columns."""
        df = pd.DataFrame({
            "id": [1.0, 2.0, 3.0],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })
        path = os.path.join(self.tmpdir, "mixed.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(meta.number_columns, 3)

    def test_dta_roundtrip_nan_in_middle(self):
        """DTA round-trip with NaN in middle of numeric column."""
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        path = os.path.join(self.tmpdir, "nanmid.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 3)
        self.assertTrue(np.isnan(df_read["val"].iloc[1]))

    def test_dta_roundtrip_many_rows(self):
        """DTA round-trip with many rows."""
        df = pd.DataFrame({"x": [float(i) for i in range(1000)]})
        path = os.path.join(self.tmpdir, "manyrows.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 1000)
        self.assertEqual(meta.number_rows, 1000)

    def test_dta_roundtrip_with_file_label(self):
        """DTA round-trip with file label preserves label."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "labeled.dta")
        pyreadstat.write_dta(df, path, file_label="Test DTA Label")
        _, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.file_label, "Test DTA Label")

    def test_dta_roundtrip_with_column_labels(self):
        """DTA round-trip preserves column labels."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        path = os.path.join(self.tmpdir, "collabeled.dta")
        pyreadstat.write_dta(df, path, column_labels=["X var", "Y var"])
        _, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.column_labels, ["X var", "Y var"])

    def test_dta_roundtrip_unicode_strings(self):
        """DTA round-trip with unicode characters in string data."""
        df = pd.DataFrame({"text": ["cafe\u0301", "nai\u0308ve", "re\u0301sume\u0301"]})
        path = os.path.join(self.tmpdir, "unicode.dta")
        pyreadstat.write_dta(df, path)
        df_read, _ = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 3)


class TestXPTRoundTripExtended(unittest.TestCase):
    """Extended XPT round-trip tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_xport_v5_roundtrip(self):
        """XPT v5 round-trip preserves data."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        path = os.path.join(self.tmpdir, "v5.xpt")
        pyreadstat.write_xport(df, path, file_format_version=5)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(df_read.columns.tolist(), ["X"])

    def test_xport_v8_roundtrip_multiple_cols(self):
        """XPT v8 round-trip with multiple columns."""
        df = pd.DataFrame({
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": [5.0, 6.0],
        })
        path = os.path.join(self.tmpdir, "v8multi.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, meta = pyreadstat.read_xport(path)
        df_read.columns = [c.lower() for c in df_read.columns]
        self.assertEqual(len(df_read), 2)
        self.assertEqual(meta.number_columns, 3)

    def test_xport_roundtrip_with_table_name(self):
        """XPT round-trip preserves table name."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "named.xpt")
        pyreadstat.write_xport(df, path, table_name="MYTBL", file_format_version=8)
        _, meta = pyreadstat.read_xport(path)
        self.assertEqual(meta.table_name, "MYTBL")

    def test_xport_v5_single_value(self):
        """XPT v5 with single value."""
        df = pd.DataFrame({"VAL": [42.0]})
        path = os.path.join(self.tmpdir, "single_v5.xpt")
        pyreadstat.write_xport(df, path, file_format_version=5)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 1)
        self.assertAlmostEqual(df_read["VAL"].iloc[0], 42.0)

    def test_xport_v8_nan_values(self):
        """XPT v8 round-trip with NaN values."""
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        path = os.path.join(self.tmpdir, "nan_v8.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 3)
        self.assertTrue(np.isnan(df_read.iloc[1, 0]))


class TestPORRoundTripExtended(unittest.TestCase):
    """Extended POR round-trip tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_por_roundtrip_multiple_columns(self):
        """POR round-trip with multiple numeric columns."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        path = os.path.join(self.tmpdir, "multi.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        df_read.columns = [c.lower() for c in df_read.columns]
        self.assertEqual(len(df_read), 3)
        self.assertEqual(meta.number_columns, 2)

    def test_por_roundtrip_string_column(self):
        """POR round-trip with string column."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        path = os.path.join(self.tmpdir, "str.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        self.assertEqual(len(df_read), 2)

    def test_por_roundtrip_nan_values(self):
        """POR round-trip with NaN values."""
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
        path = os.path.join(self.tmpdir, "nan.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        self.assertEqual(len(df_read), 3)

    def test_por_roundtrip_many_rows(self):
        """POR round-trip with many rows."""
        df = pd.DataFrame({"x": [float(i) for i in range(500)]})
        path = os.path.join(self.tmpdir, "manyrows.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        self.assertEqual(len(df_read), 500)


class TestSAVRoundTripExtended(unittest.TestCase):
    """Extended SAV round-trip tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_roundtrip_date_column(self):
        """SAV round-trip with date column."""
        df = pd.DataFrame({
            "mydate": [datetime(2020, 1, 1), datetime(2021, 6, 15)],
        })
        path = os.path.join(self.tmpdir, "dates.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 2)

    def test_sav_roundtrip_unicode_string(self):
        """SAV round-trip with unicode strings."""
        df = pd.DataFrame({"text": ["\u00e9l\u00e8ve", "caf\u00e9", "\u00fc\u00f1\u00ee"]})
        path = os.path.join(self.tmpdir, "unicode.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(df_read["text"].tolist(), ["\u00e9l\u00e8ve", "caf\u00e9", "\u00fc\u00f1\u00ee"])

    def test_sav_roundtrip_empty_string_values(self):
        """SAV round-trip with empty string values."""
        df = pd.DataFrame({"text": ["hello", "", "world"]})
        path = os.path.join(self.tmpdir, "emptystr.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)

    def test_sav_roundtrip_negative_values(self):
        """SAV round-trip with negative numeric values."""
        df = pd.DataFrame({"val": [-100.5, -0.001, 0.0, 100.5]})
        path = os.path.join(self.tmpdir, "negative.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertAlmostEqual(df_read["val"].iloc[0], -100.5, places=1)
        self.assertAlmostEqual(df_read["val"].iloc[3], 100.5, places=1)

    def test_sav_roundtrip_with_note_and_label(self):
        """SAV round-trip with both note and file label."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "notelabel.sav")
        pyreadstat.write_sav(df, path, file_label="My Label", note="My Note")
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.file_label, "My Label")
        self.assertIn("My Note", meta.notes)

    def test_sav_roundtrip_long_string(self):
        """SAV round-trip with very long string values."""
        long_str = "A" * 500
        df = pd.DataFrame({"text": [long_str, "short"]})
        path = os.path.join(self.tmpdir, "longstr.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertTrue(df_read["text"].iloc[0].startswith("A" * 100))

    def test_zsav_roundtrip_basic(self):
        """ZSAV (compressed SAV) round-trip."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
        path = os.path.join(self.tmpdir, "basic.zsav")
        pyreadstat.write_sav(df, path, compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(meta.number_columns, 2)

    def test_zsav_roundtrip_preserves_values(self):
        """ZSAV round-trip preserves numeric values accurately."""
        df = pd.DataFrame({"val": [3.14159, 2.71828, 1.41421]})
        path = os.path.join(self.tmpdir, "precise.zsav")
        pyreadstat.write_sav(df, path, compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        for i in range(3):
            self.assertAlmostEqual(df_read["val"].iloc[i], df["val"].iloc[i], places=4)

    def test_sav_roundtrip_variable_value_labels(self):
        """SAV round-trip preserves variable_value_labels."""
        df = pd.DataFrame({"gender": [1.0, 2.0], "status": [1.0, 0.0]})
        vvl = {
            "gender": {1.0: "Male", 2.0: "Female"},
            "status": {1.0: "Active", 0.0: "Inactive"},
        }
        path = os.path.join(self.tmpdir, "vvl.sav")
        pyreadstat.write_sav(df, path, variable_value_labels=vvl)
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.variable_value_labels["gender"], {1.0: "Male", 2.0: "Female"})
        self.assertEqual(meta.variable_value_labels["status"], {1.0: "Active", 0.0: "Inactive"})


class TestWriteReadWithDates(unittest.TestCase):
    """Tests for write/read with various date types."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_datetime_roundtrip(self):
        """SAV round-trip with datetime column."""
        df = pd.DataFrame({
            "dt": [datetime(2020, 1, 1, 12, 30, 0), datetime(2021, 12, 31, 23, 59, 59)]
        })
        path = os.path.join(self.tmpdir, "dt.sav")
        pyreadstat.write_sav(df, path)
        df_read, _ = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 2)

    def test_dta_datetime_roundtrip(self):
        """DTA round-trip with datetime column."""
        df = pd.DataFrame({
            "dt": [datetime(2020, 1, 1), datetime(2021, 6, 15)]
        })
        path = os.path.join(self.tmpdir, "dt.dta")
        pyreadstat.write_dta(df, path)
        df_read, _ = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 2)


if __name__ == "__main__":
    unittest.main()
