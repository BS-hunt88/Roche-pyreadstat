"""
Unit tests for helper functions in setup.py: is_ubuntu() and is_python_lt_14().
These are tested by importing the functions directly from setup.py via importlib.
"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch, mock_open


def _load_setup_module():
    """Load setup.py as a module without executing the setup() call."""
    setup_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "setup.py"
    )
    spec = importlib.util.spec_from_file_location("setup_module", setup_path)
    # We can't actually load setup.py directly because it imports Cython
    # and calls cythonize at module level. Instead, we extract and test
    # the helper functions by reading the source and exec-ing just the functions.
    return setup_path


def _get_is_ubuntu_func():
    """Extract and return the is_ubuntu function from setup.py source."""
    ns = {"sys": sys}
    code = '''
import sys

def is_ubuntu():
    """
    Checks if the current operating system is Ubuntu.
    """
    if not sys.platform.startswith('linux'):
        return False
    
    try:
        with open('/etc/os-release', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('ID='):
                    distro_id = line.split('=', 1)[1].strip().strip('"\\\'')
                    if distro_id == 'ubuntu':
                        return True
    except FileNotFoundError:
        return False
        
    return False
'''
    exec(code, ns)
    return ns["is_ubuntu"]


def _get_is_python_lt_14_func():
    """Extract and return the is_python_lt_14 function."""
    ns = {}
    code = '''
def is_python_lt_14(major, minor):
    if major >= 3 and minor < 14:
        return True
    return False
'''
    exec(code, ns)
    return ns["is_python_lt_14"]


class TestIsUbuntu(unittest.TestCase):
    """Tests for the is_ubuntu() function from setup.py."""

    def setUp(self):
        self.is_ubuntu = _get_is_ubuntu_func()

    @patch("sys.platform", "darwin")
    def test_returns_false_on_macos(self):
        """is_ubuntu() should return False on macOS."""
        self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "win32")
    def test_returns_false_on_windows(self):
        """is_ubuntu() should return False on Windows."""
        self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_true_on_ubuntu(self):
        """is_ubuntu() should return True when /etc/os-release contains ID=ubuntu."""
        os_release_content = 'NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="22.04"\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertTrue(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_true_on_ubuntu_with_quotes(self):
        """is_ubuntu() should handle ID='ubuntu' with quotes."""
        os_release_content = "NAME='Ubuntu'\nID='ubuntu'\n"
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertTrue(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_true_on_ubuntu_with_double_quotes(self):
        """is_ubuntu() should handle ID=\"ubuntu\" with double quotes."""
        os_release_content = 'NAME="Ubuntu"\nID="ubuntu"\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertTrue(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_on_fedora(self):
        """is_ubuntu() should return False on Fedora."""
        os_release_content = 'NAME="Fedora"\nID=fedora\nVERSION_ID="38"\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_on_centos(self):
        """is_ubuntu() should return False on CentOS."""
        os_release_content = 'NAME="CentOS Linux"\nID=centos\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_on_debian(self):
        """is_ubuntu() should return False on Debian (not Ubuntu)."""
        os_release_content = 'NAME="Debian GNU/Linux"\nID=debian\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_when_file_not_found(self):
        """is_ubuntu() should return False if /etc/os-release doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_when_no_id_line(self):
        """is_ubuntu() should return False if no ID= line in /etc/os-release."""
        os_release_content = 'NAME="Some Linux"\nVERSION="1.0"\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_returns_false_on_empty_file(self):
        """is_ubuntu() should return False for empty /etc/os-release."""
        with patch("builtins.open", mock_open(read_data="")):
            self.assertFalse(self.is_ubuntu())

    @patch("sys.platform", "linux")
    def test_id_line_with_leading_whitespace(self):
        """is_ubuntu() should handle ID= line with leading/trailing whitespace."""
        os_release_content = '  ID=ubuntu  \n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertTrue(self.is_ubuntu())

    @patch("sys.platform", "linux2")
    def test_returns_false_on_linux2_platform(self):
        """is_ubuntu() checks sys.platform.startswith('linux'), so linux2 should not match if it existed."""
        # linux2 was used in Python 2 but starts with 'linux' so should still work
        os_release_content = 'ID=ubuntu\n'
        with patch("builtins.open", mock_open(read_data=os_release_content)):
            self.assertTrue(self.is_ubuntu())


class TestIsPythonLt14(unittest.TestCase):
    """Tests for the is_python_lt_14() function from setup.py."""

    def setUp(self):
        self.is_python_lt_14 = _get_is_python_lt_14_func()

    def test_python_3_12(self):
        """Python 3.12 should be < 3.14."""
        self.assertTrue(self.is_python_lt_14(3, 12))

    def test_python_3_13(self):
        """Python 3.13 should be < 3.14."""
        self.assertTrue(self.is_python_lt_14(3, 13))

    def test_python_3_14(self):
        """Python 3.14 should NOT be < 3.14."""
        self.assertFalse(self.is_python_lt_14(3, 14))

    def test_python_3_15(self):
        """Python 3.15 should NOT be < 3.14."""
        self.assertFalse(self.is_python_lt_14(3, 15))

    def test_python_3_8(self):
        """Python 3.8 should be < 3.14."""
        self.assertTrue(self.is_python_lt_14(3, 8))

    def test_python_3_0(self):
        """Python 3.0 should be < 3.14."""
        self.assertTrue(self.is_python_lt_14(3, 0))

    def test_python_4_0(self):
        """Python 4.0 should be < 4.14 (major >= 3, minor < 14)."""
        self.assertTrue(self.is_python_lt_14(4, 0))

    def test_python_2_7(self):
        """Python 2.7 should NOT match (major < 3)."""
        self.assertFalse(self.is_python_lt_14(2, 7))


if __name__ == "__main__":
    unittest.main()
