class ServiceDependencyError(RuntimeError):
    """Raised when a request depends on a runtime package that is not installed."""
