class UnsupportedMultiAgentIsaacLabAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Multi-agent Isaac Lab environments are not supported in the first integration pass."
        )
