from .provider import RushDBMemoryProvider


def register(ctx):
    """Hermes entry-point registration hook."""
    ctx.register_memory_provider(RushDBMemoryProvider())


__all__ = ["RushDBMemoryProvider", "register"]
