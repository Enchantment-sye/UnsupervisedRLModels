from __future__ import annotations

import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter


def setup_logger(agent, log_dir: str) -> None:
    """Initialize agent.logger and agent.writer.

    Keeps behavior compatible with the original in `metra.py`, but adds a small guard
    to avoid duplicate FileHandlers when `setup_logger` is called multiple times.
    """
    tabular_log_file = os.path.join(log_dir, "progress.csv")  # kept for backward-compat
    _ = tabular_log_file
    text_log_file = os.path.join(log_dir, "debug.log")
    tb_dir = os.path.join(log_dir, "tb")

    agent.writer = SummaryWriter(tb_dir)

    logging.basicConfig(level=logging.INFO)
    agent.logger = logging.getLogger("DRQ_METRAAgent")

    # Guard against duplicate handlers.
    already = False
    for h in agent.logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(text_log_file):
            already = True
            break
    if not already:
        handler = logging.FileHandler(text_log_file)
        agent.logger.addHandler(handler)

    print(f"Logging to {log_dir}")


def log_diagnostics(agent, pause_for_plot: bool = False) -> None:
    # `pause_for_plot` kept for signature compatibility.
    _ = pause_for_plot
    total_time = (time.time() - agent._start_time)
    agent.logger.info("Time %.2f s" % total_time)
    epoch_time = (time.time() - agent._itr_start_time)
    agent.logger.info("EpochTime %.2f s" % epoch_time)
    agent.writer.add_scalar("TotalEnvSteps", agent.total_env_steps, agent.total_epoch)
    agent.writer.add_scalar("TotalEpoch", agent.total_epoch, agent.total_epoch)
    agent.writer.add_scalar("TimeEpoch", epoch_time, agent.total_epoch)
    agent.writer.add_scalar("TimeTotal", total_time, agent.total_epoch)
    agent.writer.flush()


def eval_log_diagnostics(agent) -> None:
    total_time = (time.time() - agent._start_time)
    agent.writer.add_scalar("eval/TotalEnvSteps", agent.total_env_steps, agent.step_itr)
    agent.writer.add_scalar("eval/TotalEpoch", agent.total_epoch, agent.step_itr)
    agent.writer.add_scalar("eval/TimeTotal", total_time, agent.step_itr)
    agent.writer.flush()


def plot_log_diagnostics(agent) -> None:
    agent.writer.add_scalar("plot/TotalEnvSteps", agent.total_env_steps, agent.step_itr)
    agent.writer.add_scalar("plot/TotalEpoch", agent.total_epoch, agent.step_itr)
    agent.writer.flush()
