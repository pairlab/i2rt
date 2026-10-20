"""Endpoint calibration with an in-memory motor chain and virtual clock."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from i2rt.robots import utils


class MotorChain:
    def __init__(self, direction=1, fail_read=False):
        self.motor_direction = np.array([1] * 6 + [direction])
        self.position = 0.5
        self.initial_torque = np.arange(7) * 0.05 + 0.1
        self.commands = []
        self.fail_read = fail_read

    def read_states(self):
        if self.fail_read and self.commands:
            raise RuntimeError("feedback lost")
        return [
            SimpleNamespace(pos=self.position if i == 6 else 0.0, eff=torque)
            for i, torque in enumerate(self.initial_torque)
        ]

    def set_commands(self, *, torques):
        self.commands.append(torques.copy())
        self.position = float(np.clip(self.position + np.sign(torques[6]), -2.0, 3.0))


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(utils.time, "time", lambda: self.now)
        self.sleep = patch.object(utils.time, "sleep", self.advance)
        self.clock.start()
        self.sleep.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(self.sleep.stop)

    def advance(self, seconds):
        self.now += seconds

    def test_known_stroke_probes_only_open_endpoint_and_preserves_arm_torques(self):
        for direction, open_position in [(1, -2.0), (-1, 3.0)]:
            with self.subTest(direction=direction):
                chain = MotorChain(direction)
                limits = utils.detect_gripper_limits(chain, motor_stroke=6.57)
                np.testing.assert_allclose(limits, [open_position + direction * 6.57, open_position])
                self.assertTrue(all(command[6] * direction < 0 for command in chain.commands[:-1]))
                self.assertEqual(chain.commands[-1][6], 0)
                for command in chain.commands:
                    np.testing.assert_array_equal(command[:6], chain.initial_torque[:6])

    def test_feedback_failure_releases_gripper_seeking_torque(self):
        chain = MotorChain(fail_read=True)
        with self.assertRaisesRegex(RuntimeError, "feedback lost"):
            utils.detect_gripper_limits(chain, motor_stroke=6.57)
        self.assertEqual(chain.commands[-1][6], 0)
        np.testing.assert_array_equal(chain.commands[-1][:6], chain.initial_torque[:6])

    def test_invalid_stroke_issues_no_commands(self):
        for stroke in [0.0, -1.0, np.nan, np.inf]:
            with self.subTest(stroke=stroke):
                chain = MotorChain()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    utils.detect_gripper_limits(chain, motor_stroke=stroke)
                self.assertFalse(chain.commands)

    def test_unknown_stroke_preserves_two_endpoint_calibration(self):
        chain = MotorChain()
        limits = utils.detect_gripper_limits(chain)
        np.testing.assert_allclose(limits, [3.0, -2.0])
        self.assertTrue(any(command[6] > 0 for command in chain.commands))
        self.assertTrue(any(command[6] < 0 for command in chain.commands))
        self.assertEqual(chain.commands[-1][6], 0)


if __name__ == "__main__":
    unittest.main()
