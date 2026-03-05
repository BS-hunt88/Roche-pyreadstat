"""
Unit tests for pyreadstat/pyfunctions.py edge cases.

Tests set_value_labels() and set_catalog_to_sas() with various edge cases
including empty metadata, missing columns, different format options, etc.
"""

import unittest
import warnings
from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

import pyreadstat
from pyreadstat.pyfunctions import set_value_labels, set_catalog_to_sas


class MockMetadata:
    """Mock metadata object that mimics pyreadstat metadata_container."""

    def __init__(self, value_labels=None, variable_to_label=None,
                 variable_value_labels=None):
        self.value_labels = value_labels
        self.variable_to_label = variable_to_label
        self.variable_value_labels = variable_value_labels or {}


class TestSetValueLabelsEdgeCases(unittest.TestCase):
    """Tests for set_value_labels() edge cases."""

    def test_empty_value_labels(self):
        """When metadata has no value_labels, dataframe should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        meta = MockMetadata(value_labels={}, variable_to_label={"a": "fmt_a"})
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_none_value_labels(self):
        """When metadata.value_labels is None, dataframe should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(value_labels=None, variable_to_label={"a": "fmt_a"})
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_empty_variable_to_label(self):
        """When metadata has no variable_to_label, dataframe should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(
            value_labels={"fmt_a": {1: "one", 2: "two", 3: "three"}},
            variable_to_label={},
        )
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_none_variable_to_label(self):
        """When metadata.variable_to_label is None, dataframe should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(
            value_labels={"fmt_a": {1: "one", 2: "two", 3: "three"}},
            variable_to_label=None,
        )
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_both_empty(self):
        """When both metadata fields are empty, dataframe should be returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(value_labels={}, variable_to_label={})
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_variable_not_in_dataframe(self):
        """When variable_to_label references a column not in df, it should be skipped."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(
            value_labels={"fmt_b": {1: "one"}},
            variable_to_label={"nonexistent_col": "fmt_b"},
        )
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_label_name_not_in_value_labels(self):
        """When variable_to_label references a label name not in value_labels."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        meta = MockMetadata(
            value_labels={"fmt_a": {1: "one"}},
            variable_to_label={"a": "nonexistent_fmt"},
        )
        result = set_value_labels(df, meta)
        self.assertTrue(result.equals(df))

    def test_basic_label_replacement(self):
        """Basic value label replacement should work."""
        df = pd.DataFrame({"gender": [1.0, 2.0, 1.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        expected = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
        self.assertTrue(result.equals(expected))

    def test_formats_as_category_true(self):
        """With formats_as_category=True, replaced columns should be categorical."""
        df = pd.DataFrame({"gender": [1.0, 2.0, 1.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=True)
        self.assertTrue(result["gender"].dtype.name == "category")

    def test_formats_as_category_false(self):
        """With formats_as_category=False, replaced columns should NOT be categorical."""
        df = pd.DataFrame({"gender": [1.0, 2.0, 1.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertFalse(result["gender"].dtype.name == "category")

    def test_formats_as_ordered_category(self):
        """With formats_as_ordered_category=True, columns should be ordered categorical."""
        df = pd.DataFrame({"level": [1.0, 2.0, 3.0]})
        meta = MockMetadata(
            value_labels={"level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"}},
            variable_to_label={"level": "level_fmt"},
        )
        result = set_value_labels(
            df, meta, formats_as_ordered_category=True
        )
        self.assertTrue(result["level"].dtype.name == "category")
        self.assertTrue(result["level"].cat.ordered)

    def test_ordered_category_overrides_category(self):
        """formats_as_ordered_category should take precedence over formats_as_category."""
        df = pd.DataFrame({"level": [1.0, 2.0, 3.0]})
        meta = MockMetadata(
            value_labels={"level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"}},
            variable_to_label={"level": "level_fmt"},
        )
        result = set_value_labels(
            df, meta, formats_as_category=True, formats_as_ordered_category=True
        )
        self.assertTrue(result["level"].cat.ordered)

    def test_values_not_in_labels_default_to_self(self):
        """Values in the dataframe not present in labels should map to themselves."""
        df = pd.DataFrame({"status": [1.0, 2.0, 99.0]})
        meta = MockMetadata(
            value_labels={"status_fmt": {1.0: "Active", 2.0: "Inactive"}},
            variable_to_label={"status": "status_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        # 99.0 should remain as 99.0 (mapped to itself)
        self.assertEqual(result["status"].iloc[0], "Active")
        self.assertEqual(result["status"].iloc[1], "Inactive")
        self.assertEqual(result["status"].iloc[2], 99.0)

    def test_returns_copy_not_original(self):
        """set_value_labels should return a copy, not modify the original."""
        df = pd.DataFrame({"gender": [1.0, 2.0]})
        original_values = df["gender"].tolist()
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        # Original should be unchanged
        self.assertEqual(df["gender"].tolist(), original_values)
        # Result should be different
        self.assertNotEqual(result["gender"].tolist(), original_values)

    def test_multiple_columns_with_labels(self):
        """Multiple columns can have different label formats applied."""
        df = pd.DataFrame({"gender": [1.0, 2.0], "level": [1.0, 3.0]})
        meta = MockMetadata(
            value_labels={
                "gender_fmt": {1.0: "Male", 2.0: "Female"},
                "level_fmt": {1.0: "low", 2.0: "medium", 3.0: "high"},
            },
            variable_to_label={"gender": "gender_fmt", "level": "level_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertEqual(result["gender"].iloc[0], "Male")
        self.assertEqual(result["level"].iloc[1], "high")

    def test_partial_columns_with_labels(self):
        """Only columns with matching labels should be transformed."""
        df = pd.DataFrame({"gender": [1.0, 2.0], "score": [10.0, 20.0]})
        meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}},
            variable_to_label={"gender": "gender_fmt"},
        )
        result = set_value_labels(df, meta, formats_as_category=False)
        self.assertEqual(result["gender"].iloc[0], "Male")
        # score should be unchanged
        self.assertEqual(result["score"].iloc[0], 10.0)


class TestSetCatalogToSas(unittest.TestCase):
    """Tests for set_catalog_to_sas() edge cases."""

    def test_no_catalog_value_labels(self):
        """When catalog has no value_labels, returns copy of original data."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"a": "fmt_a"},
        )
        catalog_meta = MockMetadata(value_labels=None)
        result_df, result_meta = set_catalog_to_sas(df, sas_meta, catalog_meta)
        self.assertTrue(result_df.equals(df))

    def test_empty_catalog_value_labels(self):
        """When catalog has empty value_labels dict, returns copy of original data."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"a": "fmt_a"},
        )
        catalog_meta = MockMetadata(value_labels={})
        result_df, result_meta = set_catalog_to_sas(df, sas_meta, catalog_meta)
        self.assertTrue(result_df.equals(df))

    def test_no_sas_variable_to_label(self):
        """When sas_metadata has no variable_to_label, returns copy of original data."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label=None,
        )
        catalog_meta = MockMetadata(
            value_labels={"fmt_a": {1.0: "one", 2.0: "two"}}
        )
        result_df, result_meta = set_catalog_to_sas(df, sas_meta, catalog_meta)
        self.assertTrue(result_df.equals(df))

    def test_empty_sas_variable_to_label(self):
        """When sas_metadata has empty variable_to_label, returns copy of original data."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={},
        )
        catalog_meta = MockMetadata(
            value_labels={"fmt_a": {1.0: "one", 2.0: "two"}}
        )
        result_df, result_meta = set_catalog_to_sas(df, sas_meta, catalog_meta)
        self.assertTrue(result_df.equals(df))

    def test_catalog_applied_successfully(self):
        """When both catalog and sas metadata have required fields, labels are applied."""
        df = pd.DataFrame({"gender": [1.0, 2.0]})
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
        self.assertEqual(result_df["gender"].iloc[0], "Male")
        self.assertEqual(result_df["gender"].iloc[1], "Female")

    def test_catalog_sets_variable_value_labels(self):
        """Catalog application should populate variable_value_labels in metadata."""
        df = pd.DataFrame({"gender": [1.0, 2.0]})
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
        self.assertEqual(
            result_meta.variable_value_labels["gender"],
            {1.0: "Male", 2.0: "Female"},
        )

    def test_catalog_does_not_modify_originals(self):
        """set_catalog_to_sas should not modify the original metadata or dataframe."""
        df = pd.DataFrame({"gender": [1.0, 2.0]})
        sas_meta = MockMetadata(
            value_labels=None,
            variable_to_label={"gender": "gender_fmt"},
        )
        catalog_meta = MockMetadata(
            value_labels={"gender_fmt": {1.0: "Male", 2.0: "Female"}}
        )
        original_df_values = df["gender"].tolist()
        result_df, result_meta = set_catalog_to_sas(
            df, sas_meta, catalog_meta, formats_as_category=False
        )
        # Original df should be unchanged
        self.assertEqual(df["gender"].tolist(), original_df_values)
        # Original sas_meta should not have variable_value_labels
        self.assertEqual(sas_meta.variable_value_labels, {})

    def test_catalog_with_unmatched_variable(self):
        """Variables in variable_to_label that don't match catalog labels should be skipped."""
        df = pd.DataFrame({"gender": [1.0, 2.0], "age": [25.0, 30.0]})
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
        # gender should have labels applied
        self.assertIn("gender", result_meta.variable_value_labels)
        # age should NOT have labels (age_fmt not in catalog)
        self.assertNotIn("age", result_meta.variable_value_labels)

    def test_catalog_with_ordered_category(self):
        """Catalog with formats_as_ordered_category should produce ordered categories."""
        df = pd.DataFrame({"level": [1.0, 2.0, 3.0]})
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
        self.assertTrue(result_df["level"].cat.ordered)


if __name__ == "__main__":
    unittest.main()
