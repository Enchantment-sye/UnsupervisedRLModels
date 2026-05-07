class UnsupportedNonBoxIsaacLabAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Non-Box Isaac Lab observation/action spaces are not supported in the first integration pass."
        )
