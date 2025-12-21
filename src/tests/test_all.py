import pytest
import siebren


def test_sum_as_string():
    assert siebren.sum_as_string(1, 1) == "2"
