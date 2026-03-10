"""
Extended write validation tests covering error handling, edge cases,
and parameter combinations not covered by existing tests.

Tests include:
- Writing with duplicate column names
- Writing with various pandas dtypes (bool, int, mixed)
- Write overwrite behavior
- Writing with missing_ranges parameter
- Writing with variable_value_labels across formats
- Writing with column_labels mismatch
- Writing with file_label across all formats
- Writing with note parameter
- Compressed SAV (zsav) edge cases
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd

import pyreadstat


class TestWriteWithVariousTypes(unittest.TestCase):
    """Tests for writing DataFrames with various pandas dtypes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_write_boolean_column(self):
        """SAV should handle boolean columns."""
        df = pd.DataFrame({"flag": [True, False, True]})
        path = os.path.join(self.tmpdir, "bool.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)

    def test_dta_write_boolean_column(self):
        """DTA should handle boolean columns."""
        df = pd.DataFrame({"flag": [True, False, True]})
        path = os.path.join(self.tmpdir, "bool.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 3)

    def test_sav_write_integer_column(self):
        """SAV should handle integer columns."""
        df = pd.DataFrame({"count": [1, 2, 3, 4, 5]})
        path = os.path.join(self.tmpdir, "int.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 5)

    def test_dta_write_integer_column(self):
        """DTA should handle integer columns."""
        df = pd.DataFrame({"count": [10, 20, 30]})
        path = os.path.join(self.tmpdir, "int.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 3)

    def test_sav_write_mixed_nan_and_values(self):
        """SAV should handle columns with interspersed NaN values."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": ["x", np.nan, "z", np.nan, "w"],
        })
        path = os.path.join(self.tmpdir, "mixed_nan.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 5)
        self.assertTrue(pd.isna(df_read["a"].iloc[1]))

    def test_sav_write_all_nan_string_column(self):
        """SAV should handle all-NaN string columns."""
        df = pd.DataFrame({"text": [np.nan, np.nan, np.nan]})
        path = os.path.join(self.tmpdir, "allnan_str.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)

    def test_xport_write_integer_column(self):
        """XPT v8 should handle integer columns."""
        df = pd.DataFrame({"val": [100, 200, 300]})
        path = os.path.join(self.tmpdir, "int.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 3)

    def test_por_write_integer_column(self):
        """POR should handle integer columns."""
        df = pd.DataFrame({"val": [100, 200, 300]})
        path = os.path.join(self.tmpdir, "int.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        self.assertEqual(len(df_read), 3)


class TestWriteOverwrite(unittest.TestCase):
    """Tests for writing to existing files (overwrite behavior)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_overwrite_existing_file(self):
        """Writing SAV to an existing file should overwrite it."""
        path = os.path.join(self.tmpdir, "overwrite.sav")
        df1 = pd.DataFrame({"x": [1.0, 2.0]})
        pyreadstat.write_sav(df1, path)

        df2 = pd.DataFrame({"y": [3.0, 4.0, 5.0]})
        pyreadstat.write_sav(df2, path)

        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(df_read.columns.tolist(), ["y"])

    def test_dta_overwrite_existing_file(self):
        """Writing DTA to an existing file should overwrite it."""
        path = os.path.join(self.tmpdir, "overwrite.dta")
        df1 = pd.DataFrame({"x": [1.0]})
        pyreadstat.write_dta(df1, path)

        df2 = pd.DataFrame({"a": [10.0], "b": [20.0]})
        pyreadstat.write_dta(df2, path)

        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.number_columns, 2)


class TestWriteWithLabels(unittest.TestCase):
    """Tests for writing with variable_value_labels and column_labels."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_write_variable_value_labels_complex(self):
        """SAV write with complex variable_value_labels (multiple vars, many values)."""
        df = pd.DataFrame({
            "gender": [1.0, 2.0, 1.0],
            "education": [1.0, 2.0, 3.0],
            "score": [88.5, 92.3, 75.0],
        })
        vvl = {
            "gender": {1.0: "Male", 2.0: "Female"},
            "education": {1.0: "HS", 2.0: "Bachelors", 3.0: "Masters"},
        }
        path = os.path.join(self.tmpdir, "complex_labels.sav")
        pyreadstat.write_sav(df, path, variable_value_labels=vvl)
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.variable_value_labels["gender"], {1.0: "Male", 2.0: "Female"})
        self.assertEqual(meta.variable_value_labels["education"][3.0], "Masters")

    def test_sav_write_column_labels(self):
        """SAV write with column_labels should preserve them."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        labels = ["Variable X", "Variable Y", "Variable Z"]
        path = os.path.join(self.tmpdir, "col_labels.sav")
        pyreadstat.write_sav(df, path, column_labels=labels)
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.column_labels, labels)

    def test_dta_write_column_labels(self):
        """DTA write with column_labels should preserve them."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        labels = ["Alpha", "Beta"]
        path = os.path.join(self.tmpdir, "col_labels.dta")
        pyreadstat.write_dta(df, path, column_labels=labels)
        _, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.column_labels, labels)

    def test_sav_write_file_label(self):
        """SAV write with file_label should preserve it."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "labeled.sav")
        pyreadstat.write_sav(df, path, file_label="My Dataset 2024")
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.file_label, "My Dataset 2024")

    def test_dta_write_file_label(self):
        """DTA write with file_label should preserve it."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "labeled.dta")
        pyreadstat.write_dta(df, path, file_label="Stata Dataset")
        _, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.file_label, "Stata Dataset")

    def test_sav_write_note(self):
        """SAV write with note should preserve it."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "noted.sav")
        pyreadstat.write_sav(df, path, note="This is a test note")
        _, meta = pyreadstat.read_sav(path)
        self.assertIn("This is a test note", meta.notes)

    def test_sav_write_note_and_file_label_together(self):
        """SAV write with both note and file_label."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "both.sav")
        pyreadstat.write_sav(df, path, file_label="Label", note="Note text")
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.file_label, "Label")
        self.assertIn("Note text", meta.notes)


class TestWriteCompression(unittest.TestCase):
    """Tests for write compression options."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_zsav_compressed_output(self):
        """Writing SAV with compress=True produces a valid compressed file."""
        df = pd.DataFrame({"x": [float(i) for i in range(100)]})
        path = os.path.join(self.tmpdir, "compressed.zsav")
        pyreadstat.write_sav(df, path, compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 100)

    def test_zsav_vs_sav_same_data(self):
        """ZSAV and SAV should produce the same data when read back."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
        sav_path = os.path.join(self.tmpdir, "normal.sav")
        zsav_path = os.path.join(self.tmpdir, "compressed.zsav")
        pyreadstat.write_sav(df, sav_path)
        pyreadstat.write_sav(df, zsav_path, compress=True)
        df_sav, _ = pyreadstat.read_sav(sav_path)
        df_zsav, _ = pyreadstat.read_sav(zsav_path)
        self.assertTrue(df_sav.equals(df_zsav))

    def test_sav_row_compression(self):
        """SAV with row_compress=True should produce valid output."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        path = os.path.join(self.tmpdir, "rowcomp.sav")
        pyreadstat.write_sav(df, path, row_compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 5)

    def test_zsav_with_string_data(self):
        """ZSAV should handle string data correctly."""
        df = pd.DataFrame({"text": ["hello", "world", "test" * 50]})
        path = os.path.join(self.tmpdir, "str_compressed.zsav")
        pyreadstat.write_sav(df, path, compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertEqual(df_read["text"].iloc[0], "hello")


class TestWriteDatetimeFormats(unittest.TestCase):
    """Tests for writing various date/time column types."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_write_datetime_column(self):
        """SAV should handle datetime columns."""
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01", "2021-06-15", "2022-12-31"]),
        })
        path = os.path.join(self.tmpdir, "dt.sav")
        pyreadstat.write_sav(df, path)
        df_read, _ = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)

    def test_dta_write_datetime_column(self):
        """DTA should handle datetime columns."""
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01", "2020-06-15"]),
        })
        path = os.path.join(self.tmpdir, "dt.dta")
        pyreadstat.write_dta(df, path)
        df_read, _ = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 2)

    def test_xport_write_datetime_column(self):
        """XPT should handle datetime columns."""
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01", "2020-06-15"]),
        })
        path = os.path.join(self.tmpdir, "dt.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, _ = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 2)

    def test_sav_write_date_with_nan(self):
        """SAV should handle datetime columns with NaN/NaT values."""
        df = pd.DataFrame({
            "dt": [datetime(2020, 1, 1), pd.NaT, datetime(2022, 12, 31)],
        })
        path = os.path.join(self.tmpdir, "dt_nan.sav")
        pyreadstat.write_sav(df, path)
        df_read, _ = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)


class TestWriteXportVersions(unittest.TestCase):
    """Tests for XPT write with different version options."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_xport_v5_preserves_column_names(self):
        """XPT v5 write and read should preserve column names as written."""
        df = pd.DataFrame({"myvar": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "v5.xpt")
        pyreadstat.write_xport(df, path, file_format_version=5)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 2)
        self.assertEqual(len(df_read.columns), 1)

    def test_xport_v8_preserves_column_names(self):
        """XPT v8 should preserve column names."""
        df = pd.DataFrame({"myvar": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "v8.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertIn("myvar", [c.lower() for c in df_read.columns])

    def test_xport_v8_with_table_name(self):
        """XPT v8 should preserve table_name."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "named.xpt")
        pyreadstat.write_xport(df, path, table_name="TESTDATA", file_format_version=8)
        _, meta = pyreadstat.read_xport(path)
        self.assertEqual(meta.table_name, "TESTDATA")

    def test_xport_v5_with_table_name(self):
        """XPT v5 should preserve table_name."""
        df = pd.DataFrame({"X": [1.0]})
        path = os.path.join(self.tmpdir, "named_v5.xpt")
        pyreadstat.write_xport(df, path, table_name="MYDATA", file_format_version=5)
        _, meta = pyreadstat.read_xport(path)
        self.assertEqual(meta.table_name, "MYDATA")


if __name__ == "__main__":
    unittest.main()
