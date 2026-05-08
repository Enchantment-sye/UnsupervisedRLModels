from __future__ import annotations

import atexit
import copy
import os
import pickle
import random
import sys
import traceback

import numpy as np

from envs.generic_parallel import GenericProcessTrajectoryCollector


_ACCESS = 1
_CALL = 2
_RESULT = 3
_CLOSE = 4
_EXCEPTION = 5


class KitchenProcessTrajectoryCollector(GenericProcessTrajectoryCollector):
    """Spawn-based parallel collector isolated for D4RL Kitchen envs."""

    def __init__(self, cfg, *, num_workers: int | None = None, worker_factory=None):
        self.cfg = cfg
        requested_workers = int(num_workers if num_workers is not None else getattr(cfg, "n_parallel", 1))
        self._num_workers = max(1, requested_workers)
        self._workers = []
        self._timing_totals = self._new_timing_totals()
        if worker_factory is None:
            import multiprocessing as mp

            _assert_spawn_payload_pickleable(self.cfg)
            context = mp.get_context("spawn")
            try:
                for worker_id in range(self._num_workers):
                    self._workers.append(_KitchenSpawnWorker(context, self.cfg, worker_id))
            except Exception as exc:
                self.close()
                raise RuntimeError(
                    "Kitchen parallel sampler failed to start. "
                    "Kitchen workers use spawn+EGL and do not fall back to serial; "
                    "check the worker traceback above for MuJoCo/EGL or environment setup errors."
                ) from exc
        else:
            try:
                for worker_id in range(self._num_workers):
                    self._workers.append(worker_factory(worker_id))
            except Exception:
                self.close()
                raise


class _KitchenSpawnWorker:
    def __init__(self, context, cfg, worker_id: int):
        self._closed = False
        self._conn, conn = context.Pipe()
        self._process = context.Process(
            target=_kitchen_worker_main,
            args=(conn, cfg, int(worker_id)),
        )
        atexit.register(self.close)
        self._process.start()
        try:
            conn.close()
        except OSError:
            pass
        try:
            self._receive()
        except Exception:
            self.close()
            raise

    def access(self, name):
        self._conn.send((_ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        self._conn.send((_CALL, (name, args, kwargs)))
        return self._receive

    def reset(self, blocking=False):
        promise = self.call("reset")
        return promise() if blocking else promise

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        return promise() if blocking else promise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.send((_CLOSE, None))
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            self._conn.close()
        except OSError:
            pass
        self._process.join(5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(5)

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError) as exc:
            raise RuntimeError("Lost connection to Kitchen environment worker.") from exc
        if message == _EXCEPTION:
            raise RuntimeError(f"Kitchen parallel worker failed:\n{payload}")
        if message == _RESULT:
            return payload
        raise KeyError(f"Received message of unexpected type {message}")


def _kitchen_worker_main(conn, cfg, worker_id: int):
    env = None
    try:
        _configure_kitchen_worker_runtime(worker_id)
        worker_cfg = copy.deepcopy(cfg)
        worker_cfg.seed = int(getattr(worker_cfg, "seed", 0)) + int(worker_id)
        _seed_kitchen_worker(worker_cfg.seed)

        from envs import make_env

        env = make_env(mode="train", config=worker_cfg)
        conn.send((_RESULT, None))
        while True:
            try:
                message, payload = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if message == _ACCESS:
                conn.send((_RESULT, getattr(env, payload)))
                continue
            if message == _CALL:
                name, args, kwargs = payload
                conn.send((_RESULT, getattr(env, name)(*args, **kwargs)))
                continue
            if message == _CLOSE:
                break
            raise KeyError(f"Received message of unknown type {message}")
    except Exception:
        stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
        print(f"Error in Kitchen environment process: {stacktrace}")
        try:
            conn.send((_EXCEPTION, stacktrace))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except OSError:
            pass


def _configure_kitchen_worker_runtime(worker_id: int):
    from envs.kitchen.mujoco_compat import configure_kitchen_mujoco_runtime

    configure_kitchen_mujoco_runtime()
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"metra_mpl_{os.getuid()}"))
    os.environ["METRA_KITCHEN_WORKER_ID"] = str(int(worker_id))


def _seed_kitchen_worker(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def _assert_spawn_payload_pickleable(cfg):
    try:
        pickle.dumps(cfg)
    except Exception as exc:
        raise RuntimeError(
            "Kitchen parallel sampler requires a pickleable config because it starts workers with spawn."
        ) from exc
