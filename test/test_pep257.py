#!/usr/bin/env python3

import pytest
from ament_pep257.main import main


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    assert main(argv=[]) == 0, 'Found code style errors / warnings'
