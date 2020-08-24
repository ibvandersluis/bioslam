#!/usr/bin/env python3

import pytest
from ament_xmllint.main import main


@pytest.mark.linter
@pytest.mark.xmllint
def test_xmllint():
    assert main(argv=[]) == 0, 'Found errors'
