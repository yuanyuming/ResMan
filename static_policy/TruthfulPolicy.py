from typing import Any, Dict, Optional, Union

import numpy as np
from tianshou.data import Batch
from tianshou.policy import BasePolicy


class TruthfulPolicy(BasePolicy):
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        return Batch(act=np.array([1], dtype=np.float32))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}
