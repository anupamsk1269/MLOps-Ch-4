import pytest
# Ensure this matches your file structure
from app.main import app 

@pytest.fixture
def client():
    with app.test_client() as test_client:
        yield test_client

def test_healthcheck(client):
    """Test that the home page returns a 200 status code."""
    rv = client.get('/')
    assert rv.status_code == 200