"""Backward compatible StackOverflow plugin wrapper."""

from .stackexchange import StackExchangePlugin


class Plugin(StackExchangePlugin):
    """Alias for the StackExchange plugin preset for StackOverflow."""

    def __init__(self, **kwargs) -> None:
        super().__init__(site="stackoverflow", **kwargs)
