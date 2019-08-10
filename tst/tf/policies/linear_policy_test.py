import pytest
import rl
import mock
from mock import Mock

from rl.tf.policies import LinearPolicy


class LinearPolicyTest:

    def test_init(self):
        with pytest.raises(TypeError):
            LinearPolicy()
