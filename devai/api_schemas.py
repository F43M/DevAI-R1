from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator
except Exception:  # pragma: no cover - fallback when pydantic is missing
    from .pydantic_fallback import BaseModel, Field, validator

from .config import config

PROJECT_ROOT = Path(config.CODE_ROOT).resolve()


def _validate_path(value: str) -> str:
    target = (PROJECT_ROOT / value).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        raise ValueError("Caminho de arquivo fora do projeto.")
    return value


class FileEditRequest(BaseModel):
    file: str
    line: int = Field(..., ge=1)
    content: str

    _file_validator = validator("file", allow_reuse=True)(_validate_path)

    @validator("content")
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Conteúdo vazio não permitido.")
        return v


class FileCreateRequest(BaseModel):
    file: str
    content: str = ""

    _file_validator = validator("file", allow_reuse=True)(_validate_path)


class FileDeleteRequest(BaseModel):
    file: str

    _file_validator = validator("file", allow_reuse=True)(_validate_path)


class DirRequest(BaseModel):
    path: str

    _path_validator = validator("path", allow_reuse=True)(_validate_path)


class ApplyRefactorRequest(BaseModel):
    file_path: str
    suggested_code: str

    _file_validator = validator("file_path", allow_reuse=True)(_validate_path)

    @validator("suggested_code")
    def _code_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Código sugerido vazio.")
        return v
