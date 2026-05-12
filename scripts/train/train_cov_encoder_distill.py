import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from workspace_common import configure_runtime
from core.cov_encoder.config import parse_args
from core.cov_encoder.distill import CoverageEncoderDistillTrainer


def main(argv=None):
    configure_runtime()
    cfg = parse_args(argv)
    trainer = CoverageEncoderDistillTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
