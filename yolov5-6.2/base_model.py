from typing import Any, List


class BaseModel:

    def get_labels(self):
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError
