import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with prediction pipeline mocked out."""
    import app as app_module
    from app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        app_module._prediction_pipeline = None
        yield c


class TestIndexRoute:
    def test_get_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "SummarAI" in response.text


class TestPredictRoute:
    def test_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_text_too_long_returns_422(self, client):
        response = client.post("/predict", json={"text": "x" * 8001})
        assert response.status_code == 422

    def test_word_limit_returns_422(self, client):
        import app as app_module

        class _MockPipeline:
            def predict(self, text: str, length: str = "standard") -> str:
                return "ok"

        app_module._prediction_pipeline = _MockPipeline()
        text = " ".join(["word"] * 1001)
        response = client.post("/predict", json={"text": text})
        app_module._prediction_pipeline = None
        assert response.status_code == 422
        assert "1000-word" in response.json()["detail"]

    def test_model_not_loaded_returns_503(self, client):
        response = client.post("/predict", json={"text": "Hello world, this is a test."})
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()

    def test_valid_request_with_mock_pipeline(self, client):
        import app as app_module

        class _MockPipeline:
            def predict(self, text: str, length: str = "standard") -> str:
                return f"Summary of: {text[:20]}"

        app_module._prediction_pipeline = _MockPipeline()
        response = client.post("/predict", json={"text": "This is a long conversation that needs summarizing."})
        app_module._prediction_pipeline = None

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "word_count_in" in data
        assert "word_count_out" in data
        assert len(data["summary"]) > 0

    def test_length_parameter_accepted(self, client):
        import app as app_module

        class _MockPipeline:
            def predict(self, text: str, length: str = "standard") -> str:
                return f"Brief: {text[:10]}"

        app_module._prediction_pipeline = _MockPipeline()
        for length in ["brief", "standard", "detailed"]:
            response = client.post("/predict", json={"text": "Some text to summarize here.", "length": length})
            assert response.status_code == 200
        app_module._prediction_pipeline = None

    def test_invalid_length_returns_422(self, client):
        response = client.post("/predict", json={"text": "Some text.", "length": "super-long"})
        assert response.status_code == 422

    def test_unicode_input_does_not_crash(self, client):
        import app as app_module

        class _MockPipeline:
            def predict(self, text: str, length: str = "standard") -> str:
                return "Summary"

        app_module._prediction_pipeline = _MockPipeline()
        response = client.post("/predict", json={"text": "日本語テスト 🎌 emoji test ñoño"})
        app_module._prediction_pipeline = None

        assert response.status_code == 200
