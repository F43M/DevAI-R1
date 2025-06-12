import pytest
import importlib
from pydantic import ValidationError

from devai import api_schemas as schemas


def test_validate_path_within_and_outside(tmp_path, monkeypatch):
    monkeypatch.setattr(schemas.config, 'CODE_ROOT', str(tmp_path))
    importlib.reload(schemas)

    inside = 'file.txt'
    assert schemas._validate_path(inside) == inside

    with pytest.raises(ValueError):
        schemas._validate_path('../outside.txt')


def test_file_edit_request_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(schemas.config, 'CODE_ROOT', str(tmp_path))
    importlib.reload(schemas)

    req = schemas.FileEditRequest(file='f.txt', line=1, content='hello')
    assert req.file == 'f.txt'
    with pytest.raises(ValidationError):
        schemas.FileEditRequest(file='../x', line=1, content='hello')
    with pytest.raises(ValidationError):
        schemas.FileEditRequest(file='f.txt', line=1, content=' ')


def test_other_schema_validations(tmp_path, monkeypatch):
    monkeypatch.setattr(schemas.config, 'CODE_ROOT', str(tmp_path))
    importlib.reload(schemas)

    schemas.FileCreateRequest(file='a.txt', content='hi')
    with pytest.raises(ValidationError):
        schemas.FileCreateRequest(file='../a', content='hi')

    schemas.FileDeleteRequest(file='a.txt')
    with pytest.raises(ValidationError):
        schemas.FileDeleteRequest(file='../../a')

    schemas.DirRequest(path='subdir')
    with pytest.raises(ValidationError):
        schemas.DirRequest(path='../sub')

    schemas.ApplyRefactorRequest(file_path='a.txt', diff='@@')
    with pytest.raises(ValidationError):
        schemas.ApplyRefactorRequest(file_path='a.txt', diff='   ')

