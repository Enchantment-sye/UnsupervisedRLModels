from envs.wrappers import FrameStack, TimeLimit
import envs.wrappers as wrappers


class _EnvWithoutTrainImageTensor:
    pass


def test_wrapped_missing_attr_uses_getattr_default():
    env = FrameStack(TimeLimit(_EnvWithoutTrainImageTensor(), duration=10), k=3)

    value = getattr(env, "get_train_image_tensor", lambda: None)()

    assert value is None


class _TinyAsyncEnv:
    @property
    def obs_space(self):
        return {}

    def ping(self):
        return "pong"


def test_process_async_uses_fork_without_cloudpickle_roundtrip(monkeypatch):
    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        return

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("cloudpickle should not be used with fork workers")

    monkeypatch.setattr(wrappers.cloudpickle, "dumps", _raise_if_called)
    monkeypatch.setattr(wrappers.cloudpickle, "loads", _raise_if_called)

    def _construct():
        return _TinyAsyncEnv()

    worker = wrappers.Async(_construct, strategy="process")
    try:
        assert worker.call("ping")() == "pong"
    finally:
        worker.close()
