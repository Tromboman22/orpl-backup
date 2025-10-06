import pytest


"""
Conftest.py

This file is used to define custom pytest configurations and fixtures.
It allows for grouping and ordering of tests based on custom markers.

The first function (pytest_collection_modifyitems) sorts the tests by
marker if it wasn't already the case in the test_file. 

The second function (pytest_runtest_protocol) formats the output of the
pytest report so that we know which marker is attributed to a test that
passes or fails.

"""


# Define all the headers and corresponding titles
# The order in which these markers are arranged determines the order of the output

GROUPS = {
    # <marker>: <TITLE>,
    "metrics": "METRICS",
    "normalization": "NORMALIZATION",
}

_printed_groups = set()


# pytest calls this function automatically and runs it at the start of each test because it is part of the conftest.py hook file


# Arguments and their contents 
# session: contains all test items that are collected at startup, 
# config: links to all hook files 
# items: is a lists containing pytest.Item objects

def pytest_collection_modifyitems(session, config, items):
    # Reorder tests: metrics first, then normalization, then the rest
    grouped_tests = []
    for marker in GROUPS:
        grouped_tests.extend([item for item in items if marker in item.keywords])

    # Any tests without a known marker go at the end
    others = [item for item in items
              if not any(marker in item.keywords for marker in GROUPS)]
    items[:] = grouped_tests + others   # used to include both marked and unmarked tests


# Arguments and their content
# item: pytest.Item
# nextitem: pytest.Item or None

def pytest_runtest_protocol(item, nextitem):
    global _printed_groups

    for marker, header in GROUPS.items():
        if marker in item.keywords and marker not in _printed_groups:
            print(f"\n\n{header}", end = '')
            _printed_groups.add(marker)
            break  # stop after the first matching group

    return None

"""
# If going for separated by modules instead:


import pytest
from pathlib import Path

_printed_modules = set()

def pytest_runtest_protocol(item, nextitem):
    global _printed_modules

    # Get the test file name (e.g., test_math_utils.py)
    module_name = Path(item.fspath).stem

    # Print header once per module
    if module_name not in _printed_modules:
        print(f"\n\n=== Module: {module_name} ===")
        _printed_modules.add(module_name)

    return None  # Continue normal test running

    
def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            print(f"✓ {report.nodeid.split('::')[-1]}")
        elif report.failed:
            print(f"✗ {report.nodeid.split('::')[-1]}")
        elif report.skipped:
            print(f"- {report.nodeid.split('::')[-1]}")

"""