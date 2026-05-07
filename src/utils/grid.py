import numpy as np

class ReversibleGrid10x10:
    """
    10x10 grid MDP with 2 actions:
      a=1: 0.9 move +x, 0.1 move +y
      a=2: 0.9 move -x, 0.1 move -y

    Boundary: toroidal wrap-around (mod 10) to keep reversibility everywhere.
    """

    def __init__(self, size=10, p_major=0.9, seed=0, start=(0, 0)):
        self.size = int(size)
        self.p_major = float(p_major)
        assert 0.0 < self.p_major < 1.0
        self.rng = np.random.default_rng(seed)
        self.start = (int(start[0]), int(start[1]))
        self.state = self.start

    def reset(self, start=None):
        if start is None:
            self.state = self.start
        else:
            self.state = (int(start[0]), int(start[1]))
        return self.state

    def step(self, action: int):
        """
        Returns: next_state (x,y)
        """
        x, y = self.state
        u = self.rng.random()

        if action == 1:
            # 0.9 along +x, 0.1 along +y
            if u < self.p_major:
                x = min((x + 1), 9 )
            else:
                y = (y + 1) % self.size

        elif action == 2:
            # 0.9 along -x, 0.1 along -y
            if u < self.p_major:
                x = (x - 1) % self.size
            else:
                y = (y - 1) % self.size

        else:
            raise ValueError("action must be 1 or 2")

        self.state = (x, y)
        return self.state

# quick sanity check
if __name__ == "__main__":
    env = ReversibleGrid10x10(seed=42)
    s = env.reset()
    for _ in range(5):
        s = env.step(1)
        print("a=1 ->", s)
    for _ in range(5):
        s = env.step(2)
        print("a=2 ->", s)
