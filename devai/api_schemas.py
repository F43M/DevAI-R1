from pathlib import Path

from pydantic import BaseModel, Field, field_validator

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

    @field_validator("file")
    @classmethod
    def _file_validator(cls, v: str) -> str:
        return _validate_path(v)

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Conteúdo vazio não permitido.")
        return v


class FileCreateRequest(BaseModel):
    file: str
    content: str = ""

    @field_validator("file")
    @classmethod
    def _file_validator(cls, v: str) -> str:
        return _validate_path(v)


class FileDeleteRequest(BaseModel):
    file: str

    @field_validator("file")
    @classmethod
    def _file_validator(cls, v: str) -> str:
        return _validate_path(v)


class DirRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _path_validator(cls, v: str) -> str:
        return _validate_path(v)


class ApplyRefactorRequest(BaseModel):
    file_path: str
    diff: str

    @field_validator("file_path")
    @classmethod
    def _file_validator(cls, v: str) -> str:
        return _validate_path(v)

    @field_validator("diff")
    @classmethod
    def _diff_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Patch vazio não permitido.")
        return v
