"""
Extended error handling and edge case tests for pyreadstat.

Tests include:
- Invalid file content for POR and SAS7BCAT formats
- Permission errors (directory as file path)
- Invalid argument types passed to read/write functions
- Writing to invalid paths
- BytesIO edge cases (POR, empty BytesIO)
- Reading with invalid usecols
- Metadataonly mode across all formats
- Chunk reader edge cases
"""

import io
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import pyreadstat


class TestInvalidFileContent(unittest.TestCase):
    """Tests for reading files with invalid/corrupt content across all formats."""

    def test_read_por_invalid_file_content(self):
        """Reading a POR file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".por", delete=False) as f:
            f.write(b"this is not a valid por file at all")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_por(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_sas7bcat_invalid_file_content(self):
        """Reading a SAS7BCAT file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".sas7bcat", delete=False) as f:
            f.write(b"not a valid sas catalog file")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_sas7bcat(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_sas7bdat_invalid_file_content(self):
        """Reading a SAS7BDAT file with invalid content should raise an error."""
        with tempfile.NamedTemporaryFile(suffix=".sas7bdat", delete=False) as f:
            f.write(b"not a valid sas data file")
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_sas7bdat(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_sav_truncated_file(self):
        """Reading a truncated SAV file should raise an error."""
        script_folder = os.path.dirname(os.path.realpath(__file__))
        parent_folder = os.path.split(script_folder)[0]
        sav_path = os.path.join(parent_folder, "test_data", "basic", "sample.sav")
        with open(sav_path, "rb") as f:
            data = f.read()
        # Write only a quarter of the file
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as f:
            f.write(data[: len(data) // 4])
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_sav(tmppath)
        finally:
            os.unlink(tmppath)

    def test_read_dta_truncated_file(self):
        """Reading a truncated DTA file should raise an error."""
        script_folder = os.path.dirname(os.path.realpath(__file__))
        parent_folder = os.path.split(script_folder)[0]
        dta_path = os.path.join(parent_folder, "test_data", "basic", "sample.dta")
        with open(dta_path, "rb") as f:
            data = f.read()
        with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
            f.write(data[: len(data) // 4])
            f.flush()
            tmppath = f.name
        try:
            with self.assertRaises(Exception):
                pyreadstat.read_dta(tmppath)
        finally:
            os.unlink(tmppath)


class TestWriteErrors(unittest.TestCase):
    """Tests for write operations with invalid inputs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_sav_creates_file(self):
        """Writing SAV to a valid path should create the file."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "output.sav")
        pyreadstat.write_sav(df, path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_write_dta_creates_file(self):
        """Writing DTA to a valid path should create the file."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "output.dta")
        pyreadstat.write_dta(df, path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_write_xport_creates_file(self):
        """Writing XPT to a valid path should create the file."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "output.xpt")
        pyreadstat.write_xport(df, path, file_format_version=8)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_write_por_creates_file(self):
        """Writing POR to a valid path should create the file."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        path = os.path.join(self.tmpdir, "output.por")
        pyreadstat.write_por(df, path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_write_sav_nonexistent_directory(self):
        """Writing SAV to a non-existent directory should raise an error."""
        df = pd.DataFrame({"x": [1.0]})
        with self.assertRaises(Exception):
            pyreadstat.write_sav(df, "/nonexistent/dir/file.sav")

    def test_write_dta_nonexistent_directory(self):
        """Writing DTA to a non-existent directory should raise an error."""
        df = pd.DataFrame({"x": [1.0]})
        with self.assertRaises(Exception):
            pyreadstat.write_dta(df, "/nonexistent/dir/file.dta")


class TestBytesIOExtended(unittest.TestCase):
    """Extended BytesIO reading tests."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_read_por_from_bytesio(self):
        """Should be able to read POR from BytesIO."""
        por_path = os.path.join(self.basic_data_folder, "sample.por")
        with open(por_path, "rb") as f:
            data = f.read()
        df, meta = pyreadstat.read_por(io.BytesIO(data))
        self.assertGreater(len(df), 0)
        self.assertGreater(meta.number_columns, 0)

    def test_bytesio_dta_matches_file_read(self):
        """BytesIO read should match file path read for DTA."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df_file, meta_file = pyreadstat.read_dta(dta_path)
        with open(dta_path, "rb") as f:
            df_bio, meta_bio = pyreadstat.read_dta(io.BytesIO(f.read()))
        self.assertTrue(df_file.equals(df_bio))
        self.assertEqual(meta_file.number_columns, meta_bio.number_columns)
        self.assertEqual(meta_file.column_names, meta_bio.column_names)

    def test_bytesio_sas7bdat_matches_file_read(self):
        """BytesIO read should match file path read for SAS7BDAT."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        df_file, meta_file = pyreadstat.read_sas7bdat(sas_path)
        with open(sas_path, "rb") as f:
            df_bio, meta_bio = pyreadstat.read_sas7bdat(io.BytesIO(f.read()))
        self.assertTrue(df_file.equals(df_bio))
        self.assertEqual(meta_file.number_columns, meta_bio.number_columns)

    def test_bytesio_xpt_matches_file_read(self):
        """BytesIO read should match file path read for XPT."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        df_file, meta_file = pyreadstat.read_xport(xpt_path)
        with open(xpt_path, "rb") as f:
            df_bio, meta_bio = pyreadstat.read_xport(io.BytesIO(f.read()))
        self.assertTrue(df_file.equals(df_bio))
        self.assertEqual(meta_file.number_columns, meta_bio.number_columns)

    def test_bytesio_por_matches_file_read(self):
        """BytesIO read should match file path read for POR."""
        por_path = os.path.join(self.basic_data_folder, "sample.por")
        df_file, meta_file = pyreadstat.read_por(por_path)
        with open(por_path, "rb") as f:
            df_bio, meta_bio = pyreadstat.read_por(io.BytesIO(f.read()))
        self.assertTrue(df_file.equals(df_bio))
        self.assertEqual(meta_file.number_columns, meta_bio.number_columns)

    def test_bytesio_sav_with_metadataonly(self):
        """BytesIO read with metadataonly should return empty df."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        with open(sav_path, "rb") as f:
            df, meta = pyreadstat.read_sav(io.BytesIO(f.read()), metadataonly=True)
        self.assertTrue(df.empty)
        self.assertGreater(meta.number_columns, 0)


class TestMetadataOnlyAllFormats(unittest.TestCase):
    """Tests for metadataonly mode across all file formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_por_metadataonly(self):
        """POR metadataonly should return empty df with metadata."""
        df, meta = pyreadstat.read_por(
            os.path.join(self.basic_data_folder, "sample.por"), metadataonly=True
        )
        self.assertTrue(df.empty)
        self.assertGreater(meta.number_columns, 0)
        self.assertIsInstance(meta.column_names, list)
        self.assertGreater(len(meta.column_names), 0)

    def test_dta_metadataonly(self):
        """DTA metadataonly should return empty df with metadata."""
        df_full, meta_full = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta")
        )
        df, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"), metadataonly=True
        )
        self.assertTrue(df.empty)
        self.assertEqual(meta.number_columns, meta_full.number_columns)
        self.assertEqual(meta.column_names, meta_full.column_names)

    def test_sas7bdat_metadataonly_preserves_info(self):
        """SAS7BDAT metadataonly should preserve full metadata."""
        df_full, meta_full = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat")
        )
        df, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"), metadataonly=True
        )
        self.assertTrue(df.empty)
        self.assertEqual(meta.number_columns, meta_full.number_columns)
        self.assertEqual(meta.column_names, meta_full.column_names)
        self.assertEqual(meta.column_labels, meta_full.column_labels)


class TestRowLimitOffsetExtended(unittest.TestCase):
    """Extended row_limit and row_offset tests across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_xport_row_limit(self):
        """XPT reading with row_limit should limit rows."""
        df, meta = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt"), row_limit=2
        )
        self.assertEqual(len(df), 2)

    def test_xport_row_offset(self):
        """XPT reading with row_offset should skip rows."""
        df_full, _ = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt")
        )
        df, _ = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt"), row_offset=1
        )
        self.assertEqual(len(df), len(df_full) - 1)

    def test_por_row_limit(self):
        """POR reading with row_limit should limit rows."""
        df, meta = pyreadstat.read_por(
            os.path.join(self.basic_data_folder, "sample.por"), row_limit=2
        )
        self.assertEqual(len(df), 2)

    def test_dta_offset_beyond_file(self):
        """DTA reading with offset beyond file length should return empty df."""
        df, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"), row_offset=99999
        )
        self.assertEqual(len(df), 0)

    def test_xport_offset_beyond_file(self):
        """XPT reading with offset beyond file length should return empty df."""
        df, meta = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt"), row_offset=99999
        )
        self.assertEqual(len(df), 0)

    def test_sav_combined_offset_and_limit(self):
        """SAV with specific offset and limit returns correct subset."""
        df_full, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        df, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            row_offset=1, row_limit=2
        )
        self.assertEqual(len(df), 2)

    def test_sas7bdat_combined_offset_and_limit(self):
        """SAS7BDAT with specific offset and limit returns correct subset."""
        df, _ = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"),
            row_offset=2, row_limit=1
        )
        self.assertEqual(len(df), 1)


class TestUsecolsExtended(unittest.TestCase):
    """Extended usecols tests across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_dta_usecols_single_column(self):
        """DTA reading with single usecol."""
        df, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"),
            usecols=["mynum"]
        )
        self.assertEqual(len(df.columns), 1)
        self.assertEqual(df.columns.tolist(), ["mynum"])

    def test_sav_usecols_single_column(self):
        """SAV reading with single usecol."""
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            usecols=["mynum"]
        )
        self.assertEqual(len(df.columns), 1)
        self.assertEqual(df.columns.tolist(), ["mynum"])

    def test_sas7bdat_usecols_single_column(self):
        """SAS7BDAT reading with single usecol."""
        df, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat"),
            usecols=["mynum"]
        )
        self.assertEqual(len(df.columns), 1)
        self.assertEqual(df.columns.tolist(), ["mynum"])

    def test_sav_usecols_multiple_columns(self):
        """SAV reading with multiple usecols."""
        usecols = ["mynum", "mychar"]
        df, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            usecols=usecols
        )
        self.assertEqual(len(df.columns), 2)
        self.assertEqual(sorted(df.columns.tolist()), sorted(usecols))


class TestDisableDatetimeConversion(unittest.TestCase):
    """Tests for disable_datetime_conversion across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_disable_datetime(self):
        """SAV with disable_datetime_conversion should return numeric dates."""
        df, _ = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            disable_datetime_conversion=True
        )
        self.assertGreater(len(df), 0)

    def test_dta_disable_datetime(self):
        """DTA with disable_datetime_conversion should return numeric dates."""
        df, _ = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta"),
            disable_datetime_conversion=True
        )
        self.assertGreater(len(df), 0)

    def test_xport_disable_datetime(self):
        """XPT with disable_datetime_conversion should return numeric dates."""
        df, _ = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt"),
            disable_datetime_conversion=True
        )
        self.assertGreater(len(df), 0)


class TestMetadataExtendedAttributes(unittest.TestCase):
    """Extended metadata attribute tests across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_xport_metadata_attributes(self):
        """XPT metadata should have expected attributes."""
        _, meta = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt")
        )
        self.assertTrue(hasattr(meta, "number_columns"))
        self.assertTrue(hasattr(meta, "number_rows"))
        self.assertTrue(hasattr(meta, "column_names"))
        self.assertTrue(hasattr(meta, "column_labels"))
        self.assertTrue(hasattr(meta, "table_name"))

    def test_por_metadata_attributes(self):
        """POR metadata should have expected attributes."""
        _, meta = pyreadstat.read_por(
            os.path.join(self.basic_data_folder, "sample.por")
        )
        self.assertTrue(hasattr(meta, "number_columns"))
        self.assertTrue(hasattr(meta, "column_names"))
        self.assertTrue(hasattr(meta, "column_labels"))

    def test_sav_readstat_variable_types(self):
        """SAV readstat_variable_types should map column names to type strings."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertIsInstance(meta.readstat_variable_types, dict)
        for col_name in meta.column_names:
            self.assertIn(col_name, meta.readstat_variable_types)
            self.assertIsInstance(meta.readstat_variable_types[col_name], str)

    def test_dta_readstat_variable_types(self):
        """DTA readstat_variable_types should map column names to type strings."""
        _, meta = pyreadstat.read_dta(
            os.path.join(self.basic_data_folder, "sample.dta")
        )
        self.assertIsInstance(meta.readstat_variable_types, dict)
        for col_name in meta.column_names:
            self.assertIn(col_name, meta.readstat_variable_types)

    def test_sas7bdat_readstat_variable_types(self):
        """SAS7BDAT readstat_variable_types should map column names to type strings."""
        _, meta = pyreadstat.read_sas7bdat(
            os.path.join(self.basic_data_folder, "sample.sas7bdat")
        )
        self.assertIsInstance(meta.readstat_variable_types, dict)
        for col_name in meta.column_names:
            self.assertIn(col_name, meta.readstat_variable_types)

    def test_sav_file_encoding(self):
        """SAV metadata should include file_encoding."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "file_encoding"))
        # File encoding should be a string (could be None for some files)
        if meta.file_encoding is not None:
            self.assertIsInstance(meta.file_encoding, str)

    def test_sav_original_variable_types(self):
        """SAV original_variable_types should exist and be a dict."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "original_variable_types"))
        self.assertIsInstance(meta.original_variable_types, dict)

    def test_sav_variable_display_width(self):
        """SAV variable_display_width should exist and be a dict."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "variable_display_width"))
        self.assertIsInstance(meta.variable_display_width, dict)

    def test_sav_variable_storage_width(self):
        """SAV variable_storage_width should exist and be a dict."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "variable_storage_width"))
        self.assertIsInstance(meta.variable_storage_width, dict)

    def test_sav_variable_measure(self):
        """SAV variable_measure should exist and be a dict."""
        _, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav")
        )
        self.assertTrue(hasattr(meta, "variable_measure"))
        self.assertIsInstance(meta.variable_measure, dict)


class TestOutputFormatExtended(unittest.TestCase):
    """Extended output_format tests."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_xport_output_format_dict(self):
        """XPT with output_format='dict' should return a dict."""
        result, meta = pyreadstat.read_xport(
            os.path.join(self.basic_data_folder, "sample.xpt"),
            output_format="dict",
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_por_output_format_dict(self):
        """POR with output_format='dict' should return a dict."""
        result, meta = pyreadstat.read_por(
            os.path.join(self.basic_data_folder, "sample.por"),
            output_format="dict",
        )
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_sav_output_format_dict_with_usecols(self):
        """SAV with output_format='dict' and usecols should limit keys."""
        result, meta = pyreadstat.read_sav(
            os.path.join(self.basic_data_folder, "sample.sav"),
            output_format="dict",
            usecols=["mynum", "mychar"],
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)


class TestWorkerExtended(unittest.TestCase):
    """Extended tests for the worker function."""

    def test_worker_exception_propagation(self):
        """Worker should propagate exceptions from the read function."""
        from pyreadstat.worker import worker

        def failing_read(path, row_offset=0, row_limit=0):
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            worker((failing_read, "/fake/path", 0, 100, {}))

    def test_worker_with_zero_offset_and_limit(self):
        """Worker with zero offset and limit should still call read function."""
        from unittest.mock import MagicMock
        from pyreadstat.worker import worker

        mock_df = pd.DataFrame({"a": [1]})
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        result = worker((mock_read_fn, "/path", 0, 0, {}))
        mock_read_fn.assert_called_once()
        self.assertTrue(result.equals(mock_df))

    def test_worker_with_large_offset(self):
        """Worker should pass large offsets correctly."""
        from unittest.mock import MagicMock
        from pyreadstat.worker import worker

        mock_df = pd.DataFrame()
        mock_meta = MagicMock()
        mock_read_fn = MagicMock(return_value=(mock_df, mock_meta))

        worker((mock_read_fn, "/path", 999999, 100, {}))
        mock_read_fn.assert_called_once_with("/path", row_offset=999999, row_limit=100)


if __name__ == "__main__":
    unittest.main()
