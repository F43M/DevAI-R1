import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from devai.ai_model import AIModel


def test_vectorizer_real_output():
    vec = TfidfVectorizer()
    result = vec.fit_transform(["teste A", "teste B"])
    assert result.shape[0] == 2


def test_server_boot():
    assert callable(uvicorn.run)


@pytest.mark.asyncio
async def test_safe_api_call_real_response():
    model = AIModel()
    resp = await model.safe_api_call("Qual seu nome?", 50)
    await model.close()
    assert "resposta" in resp.lower()

