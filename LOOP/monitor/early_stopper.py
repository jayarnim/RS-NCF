class EarlyStopper:
    def __init__(
        self,
        patience: int,
        min_delta: float,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self._set_up_components()

    def check(
        self,
        current_score,
        current_epoch,
        current_model_state,
    ):
        if current_score > self._best_score + self.min_delta:
            self._best_score = current_score
            self._best_epoch = current_epoch + 1
            self._best_model_state = current_model_state
            self._counter = 0
        else:
            self._counter += 1

        if self._counter > self.patience:
            self._stop = True

    @property
    def get_should_stop(self):
        return self._stop

    @property
    def get_best_epoch(self):
        return self._best_epoch

    @property
    def get_best_score(self):
        return self._best_score

    @property
    def get_best_model_state(self):
        return self._best_model_state

    def _set_up_components(self):
        self._best_epoch = 0
        self._best_score = -float("inf")
        self._best_model_state = None
        self._counter = 0
        self._stop = False