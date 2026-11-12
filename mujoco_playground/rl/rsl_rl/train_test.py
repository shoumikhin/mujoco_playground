# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for train.py execution."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from mujoco_playground.rl.rsl_rl import train
import torch


class LoggingWriterTest(absltest.TestCase):

  def test_loads_triton_before_the_writers(self):
    # The writers load TensorFlow, which aborts if Triton loads after it.
    order = []

    class _Module(types.ModuleType):

      def __getattr__(self, name):
        order.append(self.__name__)
        return mock.MagicMock()

    runner = mock.Mock(cfg={"logger": "tensorboard"}, remote_log_dir=None)
    runner.logger.writer = None
    runner.logger.disable_logs = False
    runner.logger.log_dir = self.enterContext(tempfile.TemporaryDirectory())
    with (
        mock.patch.dict(
            sys.modules,
            {"clu": _Module("clu"), "tensorflow": _Module("tensorflow")},
        ),
        mock.patch.object(
            train.xpu, "load_triton", side_effect=lambda: order.append("triton")
        ),
    ):
      train.OnPolicyRunnerTB._prepare_logging_writer(runner)  # pylint: disable=protected-access
    self.assertEqual(order[:2], ["triton", "clu"])


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TrainTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    flags.FLAGS.mark_as_parsed()

  @parameterized.parameters(
      "CartpoleBalance",
      "CartpoleSwingUp",
      "CartpoleBalanceVision",
      "SO101PickCube",
      "SO101PickCubeVision",
      "TrossenPickCube",
      "TrossenPickCubeVision",
      "G1WholeBodyTracking",
  )
  def test_train_one_step(self, env_id: str):
    """Test train.py runs successfully for one step."""
    tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    with flagsaver.flagsaver(
        env=env_id,
        logdir=os.path.join(tmpdir, "logdir"),
        overrides='{"max_iterations": 1, "nworld": 2, "num_steps_per_env": 8}',
        wp_kernel_cache_dir=os.path.join(tmpdir, "wp_kernel_cache"),
    ):
      train.main([])


if __name__ == "__main__":
  absltest.main()
