"""
Unit tests for edge cases in pyreadstat read/write operations.

Tests error handling, BytesIO reading, empty dataframes, metadata attributes,
boundary conditions for row_limit/row_offset, and round-trip verification.
"""

import io
import os
import tempfile
import unittest

import pandas as pd
import numpy as np

import pyreadstat


class TestReadErrorHandling(unittest.TestCase):
    """Tests for error handling when reading files."""

    def test_read_sav_nonexistent_file(self):
        """Reading a nonexistent SAV file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_sav("/nonexistent/path/file.sav")

    def test_read_sas7bdat_nonexistent_file(self):
        """Reading a nonexistent SAS7BDAT file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_sas7bdat("/nonexistent/path/file.sas7bdat")

    def test_read_dta_nonexistent_file(self):
        """Reading a nonexistent DTA file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_dta("/nonexistent/path/file.dta")

    def test_read_xport_nonexistent_file(self):
        """Reading a nonexistent XPT file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_xport("/nonexistent/path/file.xpt")

    def test_read_por_nonexistent_file(self):
        """Reading a nonexistent POR file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_por("/nonexistent/path/file.por")

    def test_read_sas7bcat_nonexistent_file(self):
        """Reading a nonexistent SAS7BCAT file should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_sas7bcat("/nonexistent/path/file.sas7bcat")

    def test_read_sav_empty_path(self):
        """Reading with an empty string path should raise an error."""
        with self.assertRaises(Exception):
            pyreadstat.read_sav("")

    def test_read_sav_invalid_file_content(self):
        """Reading a file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as f:
            f.write(b"this is not a valid sav file")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_sav(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_dta_invalid_file_content(self):
        """Reading a DTA file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
            f.write(b"not a stata file")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_dta(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_xport_invalid_file_content(self):
        """Reading an XPT file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".xpt", delete=False) as f:
            f.write(b"not an xport file")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_xport(tmppath)
        finally:
            os.unlink(tmppath)


class TestBytesIOReading(unittest.TestCase):
    """Tests for reading from BytesIO objects (file-like objects)."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_read_sav_from_bytesio(self):
        """Should be able to read SAV from BytesIO."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        with open(sav_path, "rb") as f:
            data = f.read()
        df, meta = pyreadstat.read_sav(io.BytesIO(data))
        self.assertGreater(len(df), 0)
        self.assertGreater(meta.number_columns, 0)

    def test_read_dta_from_bytesio(self):
        """Should be able to read DTA from BytesIO."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        with open(dta_path, "rb") as f:
            data = f.read()
        df, meta = pyreadstat.read_dta(io.BytesIO(data))
        self.assertGreater(len(df), 0)
        self.assertGreater(meta.number_columns, 0)

    def test_read_sas7bdat_from_bytesio(self):
        """Should be able to read SAS7BDAT from BytesIO."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        with open(sas_path, "rb") as f:
            data = f.read()
        df, meta = pyreadstat.read_sas7bdat(io.BytesIO(data))
        self.assertGreater(len(df), 0)
        self.assertGreater(meta.number_columns, 0)

    def test_read_xport_from_bytesio(self):
        """Should be able to read XPT from BytesIO."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        with open(xpt_path, "rb") as f:
            data = f.read()
        df, meta = pyreadstat.read_xport(io.BytesIO(data))
        self.assertGreater(len(df), 0)
        self.assertGreater(meta.number_columns, 0)

    def test_bytesio_sav_matches_file_read(self):
        """BytesIO read should produce same results as file path read for SAV."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_file, meta_file = pyreadstat.read_sav(sav_path)
        with open(sav_path, "rb") as f:
            df_bio, meta_bio = pyreadstat.read_sav(io.BytesIO(f.read()))
        self.assertTrue(df_file.equals(df_bio))
        self.assertEqual(meta_file.number_columns, meta_bio.number_columns)
        self.assertEqual(meta_file.number_rows, meta_bio.number_rows)
        self.assertEqual(meta_file.column_names, meta_bio.column_names)


class TestEmptyDataframeWriteRead(unittest.TestCase):
    """Tests for writing and reading empty dataframes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_read_empty_sav(self):
        """Write and read back an empty dataframe in SAV format."""
        df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="str")})
        path = os.path.join(self.tmpdir, "empty.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 0)
        self.assertEqual(meta.number_columns, 2)

    def test_write_read_empty_dta(self):
        """Write and read back an empty dataframe in DTA format."""
        df = pd.DataFrame({"x": pd.Series(dtype="float64")})
        path = os.path.join(self.tmpdir, "empty.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 0)
        self.assertEqual(meta.number_columns, 1)

    def test_write_read_empty_xport(self):
        """Write and read back an empty dataframe in XPT format."""
        df = pd.DataFrame({"x": pd.Series(dtype="float64")})
        path = os.path.join(self.tmpdir, "empty.xpt")
        pyreadstat.write_xport(df, path)
        df_read, meta = pyreadstat.read_xport(path)
        self.assertEqual(len(df_read), 0)
        self.assertEqual(meta.number_columns, 1)

    def test_write_read_empty_por(self):
        """Write and read back an empty dataframe in POR format."""
        df = pd.DataFrame({"x": pd.Series(dtype="float64")})
        path = os.path.join(self.tmpdir, "empty.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        self.assertEqual(len(df_read), 0)
        self.assertEqual(meta.number_columns, 1)


class TestWriteReadRoundTrip(unittest.TestCase):
    """Tests for various write/read round-trip edge cases."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_roundtrip_single_row(self):
        """SAV round-trip with a single row."""
        df = pd.DataFrame({"val": [42.0], "name": ["test"]})
        path = os.path.join(self.tmpdir, "single.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 1)
        self.assertEqual(df_read["val"].iloc[0], 42.0)
        self.assertEqual(df_read["name"].iloc[0], "test")

    def test_dta_roundtrip_single_row(self):
        """DTA round-trip with a single row."""
        df = pd.DataFrame({"val": [42.0], "name": ["test"]})
        path = os.path.join(self.tmpdir, "single.dta")
        pyreadstat.write_dta(df, path)
        df_read, meta = pyreadstat.read_dta(path)
        self.assertEqual(len(df_read), 1)
        self.assertEqual(df_read["val"].iloc[0], 42.0)

    def test_sav_roundtrip_special_characters(self):
        """SAV round-trip with special characters in string data."""
        df = pd.DataFrame({"text": ["hello world", "foo bar", "test 123"]})
        path = os.path.join(self.tmpdir, "special.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(df_read["text"].iloc[0], "hello world")

    def test_sav_roundtrip_all_null_numeric(self):
        """SAV round-trip with all-null numeric column."""
        df = pd.DataFrame({"val": [np.nan, np.nan, np.nan]})
        path = os.path.join(self.tmpdir, "allnan.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertTrue(df_read["val"].isna().all())

    def test_sav_roundtrip_large_numeric_values(self):
        """SAV round-trip with large numeric values."""
        df = pd.DataFrame({"big": [1e15, -1e15, 0.0]})
        path = os.path.join(self.tmpdir, "large.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertAlmostEqual(df_read["big"].iloc[0], 1e15, places=0)
        self.assertAlmostEqual(df_read["big"].iloc[1], -1e15, places=0)

    def test_sav_roundtrip_many_columns(self):
        """SAV round-trip with many columns."""
        data = {f"col_{i}": [float(i)] for i in range(50)}
        df = pd.DataFrame(data)
        path = os.path.join(self.tmpdir, "manycols.sav")
        pyreadstat.write_sav(df, path)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.number_columns, 50)
        self.assertEqual(len(df_read), 1)

    def test_xport_roundtrip_v8(self):
        """XPT v8 round-trip preserves data."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        path = os.path.join(self.tmpdir, "roundtrip.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        df_read, meta = pyreadstat.read_xport(path)
        df_read.columns = [c.lower() for c in df_read.columns]
        self.assertTrue(df_read.equals(df))

    def test_por_roundtrip(self):
        """POR round-trip preserves data."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        path = os.path.join(self.tmpdir, "roundtrip.por")
        pyreadstat.write_por(df, path)
        df_read, meta = pyreadstat.read_por(path)
        df_read.columns = [c.lower() for c in df_read.columns]
        self.assertTrue(df_read.equals(df))


class TestMetadataAttributes(unittest.TestCase):
    """Tests for comprehensive metadata attribute verification."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_metadata_attributes_exist(self):
        """SAV metadata should have all expected attributes."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "number_columns"))
        self.assertTrue(hasattr(meta, "number_rows"))
        self.assertTrue(hasattr(meta, "column_names"))
        self.assertTrue(hasattr(meta, "column_labels"))
        self.assertTrue(hasattr(meta, "value_labels"))
        self.assertTrue(hasattr(meta, "variable_to_label"))
        self.assertTrue(hasattr(meta, "notes"))
        self.assertTrue(hasattr(meta, "variable_display_width"))
        self.assertTrue(hasattr(meta, "variable_storage_width"))
        self.assertTrue(hasattr(meta, "variable_measure"))
        self.assertTrue(hasattr(meta, "readstat_variable_types"))
        self.assertTrue(hasattr(meta, "original_variable_types"))
        self.assertTrue(hasattr(meta, "file_label"))
        self.assertTrue(hasattr(meta, "file_encoding"))
        self.assertTrue(hasattr(meta, "table_name"))
        self.assertTrue(hasattr(meta, "missing_ranges"))
        self.assertTrue(hasattr(meta, "variable_value_labels"))

    def test_sav_metadata_types(self):
        """SAV metadata attributes should have correct types."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertIsInstance(meta.number_columns, int)
        self.assertIsInstance(meta.number_rows, int)
        self.assertIsInstance(meta.column_names, list)
        self.assertIsInstance(meta.column_labels, list)
        self.assertIsInstance(meta.notes, list)
        self.assertIsInstance(meta.readstat_variable_types, dict)

    def test_sas7bdat_metadata_attributes_exist(self):
        """SAS7BDAT metadata should have expected attributes."""
        df, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat")
        )
        self.assertTrue(hasattr(meta, "number_columns"))
        self.assertTrue(hasattr(meta, "number_rows"))
        self.assertTrue(hasattr(meta, "column_names"))
        self.assertTrue(hasattr(meta, "column_labels"))
        self.assertTrue(hasattr(meta, "file_label"))
        self.assertTrue(hasattr(meta, "file_encoding"))
        self.assertTrue(hasattr(meta, "table_name"))

    def test_dta_metadata_attributes_exist(self):
        """DTA metadata should have expected attributes."""
        df, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta")
        )
        self.assertTrue(hasattr(meta, "number_columns"))
        self.assertTrue(hasattr(meta, "number_rows"))
        self.assertTrue(hasattr(meta, "column_names"))
        self.assertTrue(hasattr(meta, "column_labels"))
        self.assertTrue(hasattr(meta, "readstat_variable_types"))

    def test_sav_column_names_match_dataframe(self):
        """Metadata column_names should match dataframe columns."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertEqual(meta.column_names, list(df.columns))

    def test_sav_number_rows_matches_dataframe(self):
        """Metadata number_rows should match dataframe length."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertEqual(meta.number_rows, len(df))

    def test_sav_number_columns_matches_dataframe(self):
        """Metadata number_columns should match dataframe column count."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertEqual(meta.number_columns, len(df.columns))

    def test_sav_column_labels_length(self):
        """Metadata column_labels should have same length as column_names."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertEqual(len(meta.column_labels), len(meta.column_names))

    def test_metadataonly_preserves_column_info(self):
        """Metadataonly mode should still return column information."""
        df_full, meta_full = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        df_meta, meta_only = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"), metadataonly=True
        )
        self.assertEqual(meta_full.column_names, meta_only.column_names)
        self.assertEqual(meta_full.number_columns, meta_only.number_columns)
        self.assertTrue(df_meta.empty)


class TestRowLimitOffset(unittest.TestCase):
    """Tests for boundary conditions in row_limit and row_offset."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_row_limit_zero(self):
        """Reading with row_limit=0 means no limit (returns all rows)."""
        df_full, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        df_zero, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"), row_limit=0
        )
        self.assertEqual(len(df_full), len(df_zero))

    def test_sas7bdat_row_limit_zero(self):
        """Reading SAS with row_limit=0 means no limit (returns all rows)."""
        df_full, _ = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat")
        )
        df_zero, _ = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"), row_limit=0
        )
        self.assertEqual(len(df_full), len(df_zero))

    def test_sav_row_limit_one(self):
        """Reading with row_limit=1 should return exactly one row."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"), row_limit=1
        )
        self.assertEqual(len(df), 1)

    def test_sav_offset_beyond_file(self):
        """Reading with offset beyond file length should return empty dataframe."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"), row_offset=99999
        )
        self.assertEqual(len(df), 0)

    def test_sas7bdat_offset_beyond_file(self):
        """Reading SAS with offset beyond file length should return empty dataframe."""
        df, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"), row_offset=99999
        )
        self.assertEqual(len(df), 0)

    def test_sav_row_limit_larger_than_file(self):
        """Reading with row_limit larger than file should return all rows."""
        df_full, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        df_big, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"), row_limit=99999
        )
        self.assertEqual(len(df_full), len(df_big))

    def test_dta_row_limit_and_offset(self):
        """DTA reading with both row_limit and row_offset."""
        df_full, _ = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta")
        )
        df_chunk, _ = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"),
            row_offset=1,
            row_limit=2,
        )
        self.assertEqual(len(df_chunk), 2)


class TestWriteMetadataOptions(unittest.TestCase):
    """Tests for write operations with various metadata options."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sav_write_with_file_label(self):
        """SAV write with file_label should preserve the label."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "labeled.sav")
        pyreadstat.write_sav(df, path, file_label="My Test File")
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.file_label, "My Test File")

    def test_sav_write_with_column_labels(self):
        """SAV write with column_labels should preserve them."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        col_labels = ["X Label", "Y Label"]
        path = os.path.join(self.tmpdir, "collabels.sav")
        pyreadstat.write_sav(df, path, column_labels=col_labels)
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.column_labels, col_labels)

    def test_sav_write_with_note(self):
        """SAV write with note should preserve the note."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "noted.sav")
        pyreadstat.write_sav(df, path, note="Test note content")
        _, meta = pyreadstat.read_sav(path)
        self.assertIn("Test note content", meta.notes)

    def test_dta_write_with_file_label(self):
        """DTA write with file_label should preserve the label."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "labeled.dta")
        pyreadstat.write_dta(df, path, file_label="DTA Test Label")
        _, meta = pyreadstat.read_dta(path)
        self.assertEqual(meta.file_label, "DTA Test Label")

    def test_xport_write_with_table_name(self):
        """XPT write with table_name should preserve the name."""
        df = pd.DataFrame({"x": [1.0]})
        path = os.path.join(self.tmpdir, "named.xpt")
        pyreadstat.write_xport(df, path, table_name="MYTABLE", file_format_version=8)
        _, meta = pyreadstat.read_xport(path)
        self.assertEqual(meta.table_name, "MYTABLE")

    def test_sav_write_with_variable_value_labels(self):
        """SAV write with variable_value_labels should preserve them."""
        df = pd.DataFrame({"gender": [1.0, 2.0]})
        vvl = {"gender": {1.0: "Male", 2.0: "Female"}}
        path = os.path.join(self.tmpdir, "vvl.sav")
        pyreadstat.write_sav(df, path, variable_value_labels=vvl)
        _, meta = pyreadstat.read_sav(path)
        self.assertEqual(meta.variable_value_labels, vvl)

    def test_sav_write_compressed(self):
        """Compressed SAV (zsav) should produce valid output."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = os.path.join(self.tmpdir, "compressed.zsav")
        pyreadstat.write_sav(df, path, compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertTrue(df_read["x"].tolist() == [1.0, 2.0, 3.0])

    def test_sav_write_row_compressed(self):
        """Row-compressed SAV should produce valid output."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        path = os.path.join(self.tmpdir, "rowcomp.sav")
        pyreadstat.write_sav(df, path, row_compress=True)
        df_read, meta = pyreadstat.read_sav(path)
        self.assertEqual(len(df_read), 3)
        self.assertTrue(df_read["x"].tolist() == [1.0, 2.0, 3.0])


class TestOutputFormatDict(unittest.TestCase):
    """Tests for output_format='dict' reading option."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_output_format_dict(self):
        """Reading SAV with output_format='dict' should return a dict."""
        result, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            output_format="dict",
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_sas7bdat_output_format_dict(self):
        """Reading SAS7BDAT with output_format='dict' should return a dict."""
        result, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"),
            output_format="dict",
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_dta_output_format_dict(self):
        """Reading DTA with output_format='dict' should return a dict."""
        result, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"),
            output_format="dict",
        )
        self.assertIsInstance(result, dict)

    def test_dict_keys_match_column_names(self):
        """Dict output keys should match metadata column_names."""
        result, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            output_format="dict",
        )
        self.assertEqual(sorted(result.keys()), sorted(meta.column_names))

    def test_dict_values_are_lists(self):
        """Dict output values should be lists."""
        result, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            output_format="dict",
        )
        for key, val in result.items():
            self.assertIsInstance(val, list)

    def test_dict_values_length_matches_rows(self):
        """Dict output list lengths should match number of rows."""
        result, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            output_format="dict",
        )
        for key, val in result.items():
            self.assertEqual(len(val), meta.number_rows)


if __name__ == "__main__":
    unittest.main()
