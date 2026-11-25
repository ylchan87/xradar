#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for `io.backends.iris_opt`: it must return the same data as `iris`."""

import numpy as np
import pytest

import xradar as xd
from xradar.io.backends import iris, iris_opt
from xradar.util import _get_data_file


@pytest.mark.parametrize("fixture_name", ["iris0_file", "iris1_file"])
def test_iris_opt_datatree_identical(fixture_name, request, file_or_filelike):
    path = request.getfixturevalue(fixture_name)
    with _get_data_file(path, file_or_filelike) as sigmetfile:
        reference = xd.open_datatree(sigmetfile, engine="iris")
    with _get_data_file(path, file_or_filelike) as sigmetfile:
        modified = xd.open_datatree(sigmetfile, engine="iris_opt")
    assert reference.identical(modified)


def test_iris_opt_headers_and_ray_offsets(iris0_file):
    reference = iris.IrisRawFile(iris0_file, loaddata=False)
    modified = iris_opt.IrisOptRawFile(iris0_file, loaddata=False)

    assert reference.raw_product_bhdrs == modified.raw_product_bhdrs
    assert list(reference.data) == list(modified.data)
    for sweep_number in reference.data:
        for raw_file in (reference, modified):
            raw_file.get_moment(sweep_number, "azimuth")
        offsets_reference = reference.data[sweep_number]["ray_offsets"]
        offsets_modified = modified.data[sweep_number]["ray_offsets"]
        for moment, offsets in offsets_reference.items():
            np.testing.assert_array_equal(offsets, offsets_modified[moment])


def test_iris_opt_not_discovered(iris0_file):
    assert iris_opt.IrisOptBackendEntrypoint().guess_can_open(iris0_file) is False
    assert xd.io.discover_engine(iris0_file) == "iris"
