import os
import runpy


if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "scripts", "train", "train_mass_pixel.py")
    runpy.run_path(script, run_name="__main__")
