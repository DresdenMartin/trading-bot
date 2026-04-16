import os
import importlib

# import the module under test
mod = importlib.import_module('main')


def test_safe_int_valid():
    assert mod.safe_int('10', 1) == 10


def test_safe_int_invalid():
    assert mod.safe_int('abc', 5) == 5


def test_env_bool_truthy_values(monkeypatch):
    for v in ('1', 'true', 'yes', 'y', 'on', 'True'):
        monkeypatch.setenv('X', v)
        assert mod.env_bool('X', default=False) is True


def test_env_bool_falsey_values(monkeypatch):
    for v in ('0', 'false', 'no', 'n', ''):
        monkeypatch.setenv('X', v)
        assert mod.env_bool('X', default=True) is False
