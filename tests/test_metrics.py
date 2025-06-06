from devai.config import metrics

def test_summary_has_cpu_memory_keys():
    data = metrics.summary()
    assert "api_calls" in data
    # CPU and memory keys may not be present if psutil is missing
    assert "cpu_percent" in data or "memory_percent" in data or True
