"""Application exceptions. Each carries the HTTP status the API layer should map it to."""


class AppError(Exception):
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidConfigError(AppError):
    status_code = 422


class LLMNotConfiguredError(AppError):
    status_code = 503


class DocumentNotFoundError(AppError):
    status_code = 404


class QueryNotFoundError(AppError):
    status_code = 404


class DuplicateDocumentError(AppError):
    status_code = 409


class DocumentBusyError(AppError):
    status_code = 409


class EmptyCorpusError(AppError):
    status_code = 409


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class IngestionError(AppError):
    status_code = 422
