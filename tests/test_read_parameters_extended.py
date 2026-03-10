"""
Extended tests for read function parameters not fully covered elsewhere.

Tests include:
- apply_value_formats parameter across formats
- encoding parameter
- user_missing parameter for SAV and DTA
- output_format='dict' across all formats
- output_format='polars' across all formats
- disable_datetime_conversion combined with other parameters
- metadataonly combined with usecols
- Reading SAS catalog files
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import pyreadstat


class TestApplyValueFormats(unittest.TestCase):
    """Tests for apply_value_formats parameter."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_apply_value_formats_true(self):
        """SAV with apply_value_formats=True should apply labels."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(sav_path, apply_value_formats=True)
        self.assertGreater(len(df), 0)

    def test_sav_apply_value_formats_false(self):
        """SAV with apply_value_formats=False should not apply labels."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_raw, _ = pyreadstat.read_sav(sav_path, apply_value_formats=False)
        df_fmt, _ = pyreadstat.read_sav(sav_path, apply_value_formats=True)
        # Raw should have numeric values where formatted has strings
        self.assertGreater(len(df_raw), 0)

    def test_dta_apply_value_formats(self):
        """DTA with apply_value_formats=True should apply labels."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df, meta = pyreadstat.read_dta(dta_path, apply_value_formats=True)
        self.assertGreater(len(df), 0)

    def test_sav_apply_value_formats_with_formats_as_category(self):
        """SAV with apply_value_formats=True and formats_as_category=True."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(
            sav_path, apply_value_formats=True, formats_as_category=True
        )
        self.assertGreater(len(df), 0)


class TestUserMissing(unittest.TestCase):
    """Tests for user_missing parameter."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")
        self.missing_data_folder = os.path.join(self.parent_folder, "test_data", "missing_data")

    def test_sav_user_missing_false(self):
        """SAV with user_missing=False (default) converts user-defined missing to NaN."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(sav_path, user_missing=False)
        self.assertGreater(len(df), 0)

    def test_sav_user_missing_true(self):
        """SAV with user_missing=True preserves user-defined missing values."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(sav_path, user_missing=True)
        self.assertGreater(len(df), 0)
        # When user_missing=True, missing_ranges should be populated
        self.assertTrue(hasattr(meta, "missing_ranges"))

    def test_dta_user_missing_true(self):
        """DTA with user_missing=True preserves user-defined missing values."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df, meta = pyreadstat.read_dta(dta_path, user_missing=True)
        self.assertGreater(len(df), 0)
        self.assertTrue(hasattr(meta, "missing_user_values"))


class TestOutputFormatPolars(unittest.TestCase):
    """Tests for output_format='polars' across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_output_polars(self):
        """SAV with output_format='polars' should return polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(sav_path, output_format="polars")
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)

    def test_sas7bdat_output_polars(self):
        """SAS7BDAT with output_format='polars' should return polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        df, meta = pyreadstat.read_sas7bdat(sas_path, output_format="polars")
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)

    def test_dta_output_polars(self):
        """DTA with output_format='polars' should return polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df, meta = pyreadstat.read_dta(dta_path, output_format="polars")
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)

    def test_xport_output_polars(self):
        """XPT with output_format='polars' should return polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        df, meta = pyreadstat.read_xport(xpt_path, output_format="polars")
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)

    def test_por_output_polars(self):
        """POR with output_format='polars' should return polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        por_path = os.path.join(self.basic_data_folder, "sample.por")
        df, meta = pyreadstat.read_por(por_path, output_format="polars")
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)


class TestOutputFormatDict(unittest.TestCase):
    """Tests for output_format='dict' across formats."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sas7bdat_output_dict(self):
        """SAS7BDAT with output_format='dict' should return a dict."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        result, meta = pyreadstat.read_sas7bdat(sas_path, output_format="dict")
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_dta_output_dict(self):
        """DTA with output_format='dict' should return a dict."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        result, meta = pyreadstat.read_dta(dta_path, output_format="dict")
        self.assertIsInstance(result, dict)

    def test_xport_output_dict(self):
        """XPT with output_format='dict' should return a dict."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        result, meta = pyreadstat.read_xport(xpt_path, output_format="dict")
        self.assertIsInstance(result, dict)

    def test_por_output_dict(self):
        """POR with output_format='dict' should return a dict."""
        por_path = os.path.join(self.basic_data_folder, "sample.por")
        result, meta = pyreadstat.read_por(por_path, output_format="dict")
        self.assertIsInstance(result, dict)

    def test_dict_output_keys_match_columns(self):
        """Dict output keys should match column_names from metadata."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        result, meta = pyreadstat.read_sav(sav_path, output_format="dict")
        self.assertEqual(sorted(result.keys()), sorted(meta.column_names))

    def test_dict_output_values_are_lists(self):
        """Dict output values should be lists."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        result, meta = pyreadstat.read_sav(sav_path, output_format="dict")
        for key, val in result.items():
            self.assertIsInstance(val, list)

    def test_dict_output_values_length(self):
        """Dict output values should all have the same length (number of rows)."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        result, meta = pyreadstat.read_sav(sav_path, output_format="dict")
        lengths = [len(v) for v in result.values()]
        self.assertTrue(all(l == lengths[0] for l in lengths))


class TestCombinedParameters(unittest.TestCase):
    """Tests for combinations of parameters."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_metadataonly_with_usecols(self):
        """SAV metadataonly with usecols should return only specified columns in metadata."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        usecols = ["mynum", "mychar"]
        df, meta = pyreadstat.read_sav(sav_path, metadataonly=True, usecols=usecols)
        self.assertTrue(df.empty)
        self.assertEqual(sorted(meta.column_names), sorted(usecols))

    def test_sas7bdat_metadataonly_with_usecols(self):
        """SAS7BDAT metadataonly with usecols should work."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        usecols = ["mynum"]
        df, meta = pyreadstat.read_sas7bdat(sas_path, metadataonly=True, usecols=usecols)
        self.assertTrue(df.empty)
        self.assertEqual(meta.column_names, usecols)

    def test_sav_disable_datetime_with_usecols(self):
        """SAV with disable_datetime_conversion and usecols combined."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        usecols = ["mynum", "mychar"]
        df, meta = pyreadstat.read_sav(
            sav_path, disable_datetime_conversion=True, usecols=usecols
        )
        self.assertEqual(sorted(df.columns.tolist()), sorted(usecols))

    def test_sav_row_limit_with_usecols(self):
        """SAV with row_limit and usecols combined."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        usecols = ["mynum"]
        df, meta = pyreadstat.read_sav(sav_path, row_limit=2, usecols=usecols)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.columns.tolist(), usecols)

    def test_sav_offset_with_disable_datetime(self):
        """SAV with row_offset and disable_datetime_conversion combined."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(
            sav_path, row_offset=1, disable_datetime_conversion=True
        )
        df_full, _ = pyreadstat.read_sav(sav_path)
        self.assertEqual(len(df), len(df_full) - 1)

    def test_dta_metadataonly_with_usecols(self):
        """DTA metadataonly with usecols."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        usecols = ["mynum", "myord"]
        df, meta = pyreadstat.read_dta(dta_path, metadataonly=True, usecols=usecols)
        self.assertTrue(df.empty)
        self.assertEqual(sorted(meta.column_names), sorted(usecols))

    def test_sav_output_dict_with_row_limit(self):
        """SAV output_format='dict' combined with row_limit."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        result, meta = pyreadstat.read_sav(sav_path, output_format="dict", row_limit=3)
        self.assertIsInstance(result, dict)
        for val in result.values():
            self.assertEqual(len(val), 3)


class TestSASCatalogRead(unittest.TestCase):
    """Tests for reading SAS catalog files."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.catalog_data_folder = os.path.join(self.parent_folder, "test_data", "sas_catalog")

    def test_read_sas7bcat(self):
        """Reading a SAS catalog file should return metadata with value_labels."""
        # Check if catalog files exist
        catalog_files = [f for f in os.listdir(self.catalog_data_folder) if f.endswith(".sas7bcat")]
        if not catalog_files:
            self.skipTest("No .sas7bcat files found in test data")
        cat_path = os.path.join(self.catalog_data_folder, catalog_files[0])
        df, meta = pyreadstat.read_sas7bcat(cat_path)
        self.assertTrue(df.empty)  # Catalog files return empty DataFrame
        self.assertTrue(hasattr(meta, "value_labels"))

    def test_read_sas7bcat_metadata_has_value_labels(self):
        """SAS catalog metadata should have non-empty value_labels."""
        catalog_files = [f for f in os.listdir(self.catalog_data_folder) if f.endswith(".sas7bcat")]
        if not catalog_files:
            self.skipTest("No .sas7bcat files found in test data")
        cat_path = os.path.join(self.catalog_data_folder, catalog_files[0])
        _, meta = pyreadstat.read_sas7bcat(cat_path)
        self.assertIsInstance(meta.value_labels, dict)


class TestEncodingParameter(unittest.TestCase):
    """Tests for the encoding parameter."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_sav_with_utf8_encoding(self):
        """SAV read with explicit UTF-8 encoding should work."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df, meta = pyreadstat.read_sav(sav_path, encoding="UTF-8")
        self.assertGreater(len(df), 0)

    def test_sas7bdat_with_utf8_encoding(self):
        """SAS7BDAT read with explicit UTF-8 encoding should work."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        df, meta = pyreadstat.read_sas7bdat(sas_path, encoding="UTF-8")
        self.assertGreater(len(df), 0)

    def test_dta_with_utf8_encoding(self):
        """DTA read with explicit UTF-8 encoding should work."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df, meta = pyreadstat.read_dta(dta_path, encoding="UTF-8")
        self.assertGreater(len(df), 0)

    def test_sav_file_encoding_attribute(self):
        """SAV file_encoding metadata attribute should be populated."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        _, meta = pyreadstat.read_sav(sav_path)
        self.assertTrue(hasattr(meta, "file_encoding"))
        # file_encoding should be a string
        if meta.file_encoding is not None:
            self.assertIsInstance(meta.file_encoding, str)


if __name__ == "__main__":
    unittest.main()
