import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from workspace_common import configure_runtime
from core.mass.config import parse_args
from core.mass.trainer import MassPixelTrainer


def main(argv=None):
    configure_runtime()
    cfg = parse_args(argv)
    trainer = MassPixelTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
