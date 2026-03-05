"""
Unit tests for pyreadstat/__init__.py module attributes and public API surface.

Verifies that the package exports the expected symbols and version information.
"""

import unittest

import pyreadstat


class TestModuleAttributes(unittest.TestCase):
    """Tests for pyreadstat module-level attributes."""

    def test_version_exists(self):
        """__version__ attribute should exist."""
        self.assertTrue(hasattr(pyreadstat, "__version__"))

    def test_version_is_string(self):
        """__version__ should be a string."""
        self.assertIsInstance(pyreadstat.__version__, str)

    def test_version_format(self):
        """__version__ should follow semantic versioning (X.Y.Z)."""
        parts = pyreadstat.__version__.split(".")
        self.assertEqual(len(parts), 3, f"Version '{pyreadstat.__version__}' should have 3 parts")
        for part in parts:
            self.assertTrue(part.isdigit(), f"Version part '{part}' should be numeric")

    def test_version_matches_expected(self):
        """__version__ should match the expected version."""
        self.assertEqual(pyreadstat.__version__, "1.3.3")


class TestPublicAPI(unittest.TestCase):
    """Tests that the expected public API symbols are exported."""

    def test_read_sas7bdat_exported(self):
        """read_sas7bdat should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_sas7bdat))

    def test_read_xport_exported(self):
        """read_xport should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_xport))

    def test_read_dta_exported(self):
        """read_dta should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_dta))

    def test_read_sav_exported(self):
        """read_sav should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_sav))

    def test_read_por_exported(self):
        """read_por should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_por))

    def test_read_sas7bcat_exported(self):
        """read_sas7bcat should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_sas7bcat))

    def test_write_sav_exported(self):
        """write_sav should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.write_sav))

    def test_write_dta_exported(self):
        """write_dta should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.write_dta))

    def test_write_xport_exported(self):
        """write_xport should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.write_xport))

    def test_write_por_exported(self):
        """write_por should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.write_por))

    def test_read_file_in_chunks_exported(self):
        """read_file_in_chunks should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_file_in_chunks))

    def test_read_file_multiprocessing_exported(self):
        """read_file_multiprocessing should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.read_file_multiprocessing))

    def test_set_value_labels_exported(self):
        """set_value_labels should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.set_value_labels))

    def test_set_catalog_to_sas_exported(self):
        """set_catalog_to_sas should be accessible from pyreadstat."""
        self.assertTrue(callable(pyreadstat.set_catalog_to_sas))

    def test_readstat_error_exported(self):
        """ReadstatError should be accessible from pyreadstat."""
        self.assertTrue(hasattr(pyreadstat, "ReadstatError"))

    def test_metadata_container_exported(self):
        """metadata_container should be accessible from pyreadstat."""
        self.assertTrue(hasattr(pyreadstat, "metadata_container"))


if __name__ == "__main__":
    unittest.main()
