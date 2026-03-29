def test_base():
    assert True


def test_import_module():
    import __PYTHON_PROJECT_NAME__

    assert __PYTHON_PROJECT_NAME__.WHO_AM_I == 42
