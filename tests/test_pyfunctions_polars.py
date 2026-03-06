"""
Unit tests for polars-specific branches in pyreadstat/pyfunctions.py.

These tests cover the polars-specific code paths in set_value_labels() and
set_catalog_to_sas() that are not exercised by pandas-only tests, including:
- Object dtype unique value handling (line 47)
- All-null column skip for non-pandas (line 55)
- Mixed-type / Object dtype value replacement with warning (lines 57-65)
- Unknown dtype warning path (lines 67-70)
- Ordered category with polars
- Catalog application with polars dataframes
"""

import unittest
import warnings
from copy import deepcopy

import polars as pl
import narwhals.stable.v2 as nw

from pyreadstat.pyfunctions import set_value_labels, set_catalog_to_sas


class MockMetadata:
    """Mock metadata object that mimics pyreadstat metadata_container."""

    def __init__(self, value_labels=None, variable_to_label=None,
                 variable_value_labels=None):
        self.value_labels = value_labels
        self.variable_to_label = variable_to_label
        self.variable_value_labels = variable_value_labels or {}


class TestSetValueLabelsPolars(unittest.TestCase):
    """Tests for set_value_labels() with polars DataFrames."""

    def test_basic_label_replacement_polars(self):
        """Basic value label replacement should work with polars."""
        df = pl.DataFrame({"gender": [1.0, 2.0, 1.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result["gender"].to_list(), ["Male", "Female", "Male"])

    def test_polars_formats_as_category(self):
        """With formats_as_category=True, polars should produce Categorical."""
        df = pl.DataFrame({"gender": [1.0, 2.0, 1.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=True)
        self.assertEqual(result["gender"].dtype, pl.Categorical)

    def test_polars_formats_as_ordered_category(self):
        """With formats_as_ordered_category=True, polars should produce Enum."""
        df = pl.DataFrame({"level": [1.0, 2.0, 3.0]})
        meta = MockMetadata(
            value_labels={"level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"}},
            variable_to_label={"level": "level_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_ordered_category=True)
        # Polars Enum type is used for ordered categories
        self.assertTrue(isinstance(result["level"].dtype, pl.Enum))

    def test_polars_all_null_column_skipped(self):
        """When a polars column is all null, replacement should be skipped (line 55)."""
        df = pl.DataFrame({"gender": [None, None, None]}, schema={"gender": pl.Float64})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertIsInstance(result, pl.DataFrame)
        # All values should still be null
        self.assertEqual(result["gender"].null_count(), 3)

    def test_polars_object_dtype_with_warning(self):
        """When polars has Object dtype with mixed-type labels, it should use the fallback path with warning (lines 59-65)."""
        # Create a polars Series with Object dtype using polars native API
        ser = pl.Series(name="code", values=[object(), object()], dtype=pl.Object)
        df = pl.DataFrame({"code": ser})

        obj1 = df["code"][0]
        obj2 = df["code"][1]

        meta = MockMetadata(
            value_labels={"code_fmt": {obj1: "A", obj2: "B"}},
            variable_to_label={"code": "code_fmt"},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = set_value_labels(df, meta, formats_as_category=True)
            # Should produce a RuntimeWarning about not being able to cast to category
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            self.assertTrue(len(runtime_warnings) > 0,
                            "Expected RuntimeWarning about category casting")

    def test_polars_mixed_type_values_with_warning(self):
        """When polars labels have mixed types, fallback path should trigger (lines 57-65)."""
        df = pl.DataFrame({"status": [1.0, 2.0]})
        # Mixed types in label values: some strings, some ints
        meta = MockMetadata(
            value_labels={"status_fmt": {1.0: "Active", 2.0: 999}},
            variable_to_label={"status": "status_fmt"},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = set_value_labels(df, meta, formats_as_category=True)
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            self.assertTrue(len(runtime_warnings) > 0,
                            "Expected RuntimeWarning about category casting with mixed types")

    def test_polars_mixed_type_no_category_no_warning(self):
        """When formats_as_category=False with mixed types, no warning should be emitted."""
        df = pl.DataFrame({"status": [1.0, 2.0]})
        meta = MockMetadata(
            value_labels={"status_fmt": {1.0: "Active", 2.0: 999}},
            variable_to_label={"status": "status_fmt"},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = set_value_labels(
                df, meta, formats_as_category=False, formats_as_ordered_category=False
            )
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            self.assertEqual(len(runtime_warnings), 0,
                             "No warning expected when formats_as_category=False")

    def test_polars_values_not_in_labels_default_to_self(self):
        """Values not in labels should map to themselves in polars too."""
        df = pl.DataFrame({"status": [1.0, 2.0, 99.0]})
        meta = MockMetadata(
            value_labels={"status_fmt": {1.0: "Active", 2.0: "Inactive"}},
            variable_to_label={"status": "status_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        vals = result["status"].to_list()
        self.assertEqual(vals[0], "Active")
        self.assertEqual(vals[1], "Inactive")
        # 99.0 maps to itself
        self.assertEqual(vals[2], 99.0)

    def test_polars_empty_value_labels(self):
        """When metadata has no value_labels, polars df should be returned unchanged."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(value_labels={}, variable_to_label={"a": "fmt_a"})
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_polars_none_value_labels(self):
        """When metadata.value_labels is None, polars df should be returned unchanged."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(value_labels=None, variable_to_label={"a": "fmt_a"})
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_polars_variable_not_in_dataframe(self):
        """When variable_to_label references a column not in polars df, skip it."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(
            value_labels={"fmt_b": {1: "one"}},
            variable_to_label={"nonexistent_col": "fmt_b"},
        )
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_polars_multiple_columns(self):
        """Multiple columns with different labels in polars."""
        df = pl.DataFrame({"gender": [1.0, 2.0], "level": [1.0, 3.0]})
        meta = MockMetadata(
            value_labels={
                "gender_fmt": {1.0: "Male", 2.0: "Female"},
                "level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"},
            },
            variable_to_label={"gender": "gender_fmt", "level": "level_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertEqual(result["gender"].to_list(), ["Male", "Female"])
        self.assertEqual(result["level"].to_list(), ["low", "high"])

    def test_polars_returns_polars_dataframe(self):
        """set_value_labels with polars input should return polars DataFrame."""
        df = pl.DataFrame({"gender": [1.0, 2.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertIsInstance(result, pl.DataFrame)

    def test_polars_returns_copy_not_original(self):
        """set_value_labels should return a copy, not modify the original polars df."""
        df = pl.DataFrame({"gender": [1.0, 2.0]})
        original_values = df["gender"].to_list()
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        # Original should be unchanged
        self.assertEqual(df["gender"].to_list(), original_values)

    def test_polars_partial_columns_with_labels(self):
        """Only columns with matching labels should be transformed in polars."""
        df = pl.DataFrame({"gender": [1.0, 2.0], "score": [10.0, 20.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertEqual(result["gender"].to_list(), ["Male", "Female"])
        self.assertEqual(result["score"].to_list(), [10.0, 20.0])


class TestSetCatalogToSasPolars(unittest.TestCase):
    """Tests for set_catalog_to_sas() with polars DataFrames."""

    def test_catalog_applied_polars(self):
        """Catalog labels should be applied to polars DataFrames."""
        df = pl.DataFrame({"gender": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"gender": "gender_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}}
        )
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_category=False
        )
        self.assertIsInstance(result_df, pl.DataFrame)
        self.assertEqual(result_df["gender"].to_list(), ["Male", "Female"])

    def test_catalog_no_labels_polars(self):
        """When catalog has no value_labels, polars df should be returned as copy."""
        df = pl.DataFrame({"a": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"a": "fmt_a"},
        )
        catalog_meta = MockMetadata(value_labels=None)
        result_df, result_meta = set_catalog_to_sas(df, sas_meta, catalog_meta)
        self.assertTrue(result_df.equals(df))

    def test_catalog_sets_variable_value_labels_polars(self):
        """Catalog should set variable_value_labels in metadata with polars."""
        df = pl.DataFrame({"gender": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"gender": "gender_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}}
        )
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_category=False
        )
        self.assertIn("gender", result_meta.variable_value_labels)

    def test_catalog_does_not_modify_originals_polars(self):
        """set_catalog_to_sas should not modify original polars df or metadata."""
        df = pl.DataFrame({"gender": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"gender": "gender_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}}
        )
        original_df_values = df["gender"].to_list()
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_category=False
        )
        self.assertEqual(df["gender"].to_list(), original_df_values)

    def test_catalog_with_ordered_category_polars(self):
        """Catalog with formats_as_ordered_category should produce Enum in polars."""
        df = pl.DataFrame({"level": [1.0, 2.0, 3.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"level": "level_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"}}
        )
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_ordered_category=True
        )
        self.assertTrue(isinstance(result_df["level"].dtype, pl.Enum))

    def test_catalog_with_unmatched_variable_polars(self):
        """Variables not matching catalog labels should be skipped in polars."""
        df = pl.DataFrame({"gender": [1.0, 2.0], "age": [25.0, 30.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"gender": "gender_fmt", "age": "age_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}}
        )
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_category=False
        )
        self.assertIn("gender", result_meta.variable_value_labels)
        self.assertNotIn("age", result_meta.variable_value_labels)


if __name__ == "__main__":
    unittest.main()
