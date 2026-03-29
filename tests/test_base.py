def test_base():
    assert True


def test_import_module():
    import nanodiffusion

    assert nanodiffusion.WHO_AM_I == 42
