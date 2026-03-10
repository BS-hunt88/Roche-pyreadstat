"""
Unit tests for read_file_in_chunks() and read_file_multiprocessing().

These functions are defined in the Cython module pyreadstat.pyx but are
tested here through the public API to ensure correct behavior with
various parameters, edge cases, and error conditions.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import pyreadstat


class TestReadFileInChunks(unittest.TestCase):
    """Tests for pyreadstat.read_file_in_chunks()."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_chunk_reader_sav(self):
        """Chunk reader should yield correct chunks for SAV files."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_full, _ = pyreadstat.read_sav(sav_path)
        total_rows = len(df_full)

        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, sav_path, chunksize=2
        ))
        self.assertGreater(len(chunks), 0)
        # Each chunk is (df, meta) tuple
        total_chunk_rows = sum(len(df) for df, _ in chunks)
        self.assertEqual(total_chunk_rows, total_rows)

    def test_chunk_reader_sas7bdat(self):
        """Chunk reader should yield correct chunks for SAS7BDAT files."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        df_full, _ = pyreadstat.read_sas7bdat(sas_path)
        total_rows = len(df_full)

        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sas7bdat, sas_path, chunksize=2
        ))
        total_chunk_rows = sum(len(df) for df, _ in chunks)
        self.assertEqual(total_chunk_rows, total_rows)

    def test_chunk_reader_dta(self):
        """Chunk reader should yield correct chunks for DTA files."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df_full, _ = pyreadstat.read_dta(dta_path)
        total_rows = len(df_full)

        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta, dta_path, chunksize=2
        ))
        total_chunk_rows = sum(len(df) for df, _ in chunks)
        self.assertEqual(total_chunk_rows, total_rows)

    def test_chunk_reader_xport(self):
        """Chunk reader should yield correct chunks for XPT files."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        df_full, _ = pyreadstat.read_xport(xpt_path)
        total_rows = len(df_full)

        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_xport, xpt_path, chunksize=2
        ))
        total_chunk_rows = sum(len(df) for df, _ in chunks)
        self.assertEqual(total_chunk_rows, total_rows)

    def test_chunk_reader_chunksize_larger_than_file(self):
        """When chunksize > total rows, should return exactly one chunk."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, sav_path, chunksize=10000
        ))
        self.assertEqual(len(chunks), 1)

    def test_chunk_reader_chunksize_one(self):
        """Chunksize=1 should yield one row per chunk."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_full, _ = pyreadstat.read_sav(sav_path)
        total_rows = len(df_full)

        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, sav_path, chunksize=1
        ))
        self.assertEqual(len(chunks), total_rows)
        for df, _ in chunks:
            self.assertEqual(len(df), 1)

    def test_chunk_reader_metadata_consistent(self):
        """Each chunk should have consistent metadata."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, sav_path, chunksize=2
        ))
        # All chunks should have the same column names
        first_cols = chunks[0][1].column_names
        for _, meta in chunks:
            self.assertEqual(meta.column_names, first_cols)

    def test_chunk_reader_with_usecols(self):
        """Chunk reader should respect usecols parameter."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        usecols = ["mynum", "mychar"]
        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, sav_path, chunksize=2, usecols=usecols
        ))
        for df, meta in chunks:
            self.assertEqual(sorted(df.columns.tolist()), sorted(usecols))

    def test_chunk_reader_empty_file(self):
        """Chunk reader on an empty file should yield no chunks (or one empty)."""
        df = pd.DataFrame({"x": pd.Series(dtype="float64")})
        path = os.path.join(self.tmpdir, "empty.sav")
        pyreadstat.write_sav(df, path)
        chunks = list(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, path, chunksize=10
        ))
        total_rows = sum(len(d) for d, _ in chunks)
        self.assertEqual(total_rows, 0)


class TestReadFileMultiprocessing(unittest.TestCase):
    """Tests for pyreadstat.read_file_multiprocessing()."""

    def setUp(self):
        self.script_folder = os.path.dirname(os.path.realpath(__file__))
        self.parent_folder = os.path.split(self.script_folder)[0]
        self.basic_data_folder = os.path.join(self.parent_folder, "test_data", "basic")

    def test_multiprocess_sav(self):
        """Multiprocessing read should return same data as regular read for SAV."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_regular, _ = pyreadstat.read_sav(sav_path)
        df_mp, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sav, sav_path, num_processes=2
        )
        self.assertEqual(len(df_mp), len(df_regular))
        self.assertEqual(sorted(df_mp.columns.tolist()), sorted(df_regular.columns.tolist()))

    def test_multiprocess_sas7bdat(self):
        """Multiprocessing read should work for SAS7BDAT."""
        sas_path = os.path.join(self.basic_data_folder, "sample.sas7bdat")
        df_regular, _ = pyreadstat.read_sas7bdat(sas_path)
        df_mp, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat, sas_path, num_processes=2
        )
        self.assertEqual(len(df_mp), len(df_regular))

    def test_multiprocess_dta(self):
        """Multiprocessing read should work for DTA."""
        dta_path = os.path.join(self.basic_data_folder, "sample.dta")
        df_regular, _ = pyreadstat.read_dta(dta_path)
        df_mp, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_dta, dta_path, num_processes=2
        )
        self.assertEqual(len(df_mp), len(df_regular))

    def test_multiprocess_xport(self):
        """Multiprocessing read for XPT requires num_rows since row count
        cannot be determined from XPT metadata."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        df_regular, _ = pyreadstat.read_xport(xpt_path)
        num_rows = len(df_regular)
        df_mp, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_xport, xpt_path, num_processes=2, num_rows=num_rows
        )
        self.assertEqual(len(df_mp), num_rows)

    def test_multiprocess_xport_raises_without_num_rows(self):
        """Multiprocessing read for XPT without num_rows should raise."""
        xpt_path = os.path.join(self.basic_data_folder, "sample.xpt")
        with self.assertRaises(Exception) as ctx:
            pyreadstat.read_file_multiprocessing(
                pyreadstat.read_xport, xpt_path, num_processes=2
            )
        self.assertIn("number of rows", str(ctx.exception))

    def test_multiprocess_single_process(self):
        """Multiprocessing with num_processes=1 should still work."""
        sav_path = os.path.join(self.basic_data_folder, "sample.sav")
        df_mp, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sav, sav_path, num_processes=1
        )
        df_regular, _ = pyreadstat.read_sav(sav_path)
        self.assertEqual(len(df_mp), len(df_regular))


if __name__ == "__main__":
    unittest.main()
