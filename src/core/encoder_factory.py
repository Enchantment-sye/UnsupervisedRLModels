class EncoderFactory:
    """Factory for creating encoder instances using a registration pattern."""
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(encoder_cls):
            cls._registry[name] = encoder_cls
            return encoder_cls
        return decorator

    @classmethod
    def create(cls, encoder_type, **kwargs):
        if encoder_type not in cls._registry:
            raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(cls._registry.keys())}")

        # Filter kwargs to match the constructor signature if needed,
        # but for now we assume kwargs are compatible or handled by **kwargs in constructors
        return cls._registry[encoder_type](**kwargs)
