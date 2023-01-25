import ray

@ray.remote
class ProgressActor:
    def __init__(self, progress):
        self._progress = progress

    def get(self):
        return self._progress

    def set(self, i):
        self._progress = i

    def to_value(self):
        return self._progress
