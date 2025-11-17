import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_before_all_tests():
    """
    This fixture runs once before all tests in the session.
    """

    print("Setting up pydicom-jpeg-decoder plugins")

    from pydicom_jpeg_decoder import install_plugins

    install_plugins()

    yield
