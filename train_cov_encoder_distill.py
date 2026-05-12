import os
import runpy


if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "scripts", "train", "train_cov_encoder_distill.py")
    runpy.run_path(script, run_name="__main__")
