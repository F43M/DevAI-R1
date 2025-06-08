from devai import pydantic_fallback as pf


def test_base_model_sets_attributes():
    model = pf.BaseModel(a=1, b="x")
    assert model.a == 1
    assert model.b == "x"


def test_field_returns_default():
    assert pf.Field(5) == 5


def test_validator_decorator_returns_function():
    @pf.validator("a")
    def sample(value):
        return value

    assert sample("x") == "x"
