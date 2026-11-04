"""Smooth switches on [0, inf), each a sigmoid calibrated by a tolerance and activation length."""

from typing import Protocol

import numpy as np


class Switch(Protocol):
    """A switch turning on smoothly over [0, activation_length], from tolerance to 1 - tolerance."""

    def __init__(self, activation_length: float, tolerance: float):
        """Calibrate the sigmoid to value tolerance at 0 and 1 - tolerance at the activation length.

        Args:
            activation_length: Length from value tolerance to value 1 - tolerance.
            tolerance: Value at 0, and deficit from 1 at the activation length.

        Raises:
            ValueError: If the activation length is not positive, or the tolerance is not
                in (0, 0.5).
        """
        if not activation_length > 0:
            raise ValueError(f"activation_length must be positive, got {activation_length}")
        if not 0 < tolerance < 0.5:
            raise ValueError(f"tolerance must be in (0, 0.5), got {tolerance}")
        self._length = activation_length
        self._tolerance = tolerance
        self._span = self.logit(1.0 - tolerance) - self.logit(tolerance)

    @property
    def activation_length(self):
        """Length from value tolerance to value 1 - tolerance."""
        return self._length

    @property
    def tolerance(self):
        """Value at 0, and deficit from 1 at the activation length."""
        return self._tolerance

    def value(self, x):
        """Value of the switch.

        Args:
            x: Points to evaluate at.
        """
        return self.sigmoid(self._map_input(x))

    def derivative(self, x):
        """Derivative of the switch.

        Args:
            x: Points to evaluate at.
        """
        return self.sigmoid_derivative(self._map_input(x)) * self._span / self._length

    def _map_input(self, x):
        """Affine map taking [0, activation_length] onto [logit(tolerance), logit(1 - tolerance)].

        Args:
            x: Points to evaluate at, anywhere on the real line.
        """
        return (x / self._length - 0.5) * self._span

    @staticmethod
    def sigmoid(x):
        """Monotonically increasing map from (-inf, inf) to [0, 1], odd about (0, 1/2).

        Args:
            x: Points to evaluate at.
        """
        raise NotImplementedError

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of `sigmoid`.

        Args:
            x: Points to evaluate at.
        """
        raise NotImplementedError

    @staticmethod
    def logit(x):
        """Inverse of `sigmoid`.

        Args:
            x: Value in (0, 1).
        """
        raise NotImplementedError


class ExponentialSwitch(Switch):
    """The sigmoid 1 / (1 + exp(-x)), with tail exp(-x)."""

    @staticmethod
    def sigmoid(x):
        """Monotonically increasing map from (-inf, inf) to [0, 1], odd about (0, 1/2).

        Args:
            x: Points to evaluate at.
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of `sigmoid`.

        Args:
            x: Points to evaluate at.
        """
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)

    @staticmethod
    def logit(x):
        """Inverse of `sigmoid`.

        Args:
            x: Value in (0, 1).
        """
        return np.log(x / (1.0 - x))


class AlgebraicSwitch(Switch):
    """The sigmoid (1 + x / sqrt(1 + x^2)) / 2, with tail 1 / (4 x^2)."""

    @staticmethod
    def sigmoid(x):
        """Monotonically increasing map from (-inf, inf) to [0, 1], odd about (0, 1/2).

        Args:
            x: Points to evaluate at.
        """
        return 0.5 * (1.0 + x / np.sqrt(1.0 + x * x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of `sigmoid`.

        Args:
            x: Points to evaluate at.
        """
        return 0.5 * (1.0 + x * x) ** -1.5

    @staticmethod
    def logit(x):
        """Inverse of `sigmoid`.

        Args:
            x: Value in (0, 1).
        """
        return (2.0 * x - 1.0) / (2.0 * np.sqrt(x * (1.0 - x)))


class ArctangentSwitch(Switch):
    """The sigmoid 1/2 + arctan(x) / pi, with tail 1 / (pi x)."""

    @staticmethod
    def sigmoid(x):
        """Monotonically increasing map from (-inf, inf) to [0, 1], odd about (0, 1/2).

        Args:
            x: Points to evaluate at.
        """
        return 0.5 + np.arctan(x) / np.pi

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of `sigmoid`.

        Args:
            x: Points to evaluate at.
        """
        return 1.0 / (np.pi * (1.0 + x * x))

    @staticmethod
    def logit(x):
        """Inverse of `sigmoid`.

        Args:
            x: Value in (0, 1).
        """
        return np.tan(np.pi * (x - 0.5))


class CubicSwitch(Switch):
    """The cubic 3u^2 - 2u^3 with u = (1 + x) / 2 clipped to [0, 1], exactly 1 from x = 1 on."""

    @staticmethod
    def sigmoid(x):
        """Monotonically increasing map from (-inf, inf) to [0, 1], odd about (0, 1/2).

        Args:
            x: Points to evaluate at.
        """
        u = np.clip(0.5 * (1.0 + x), 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of `sigmoid`.

        Args:
            x: Points to evaluate at.
        """
        u = np.clip(0.5 * (1.0 + x), 0.0, 1.0)
        return 3.0 * u * (1.0 - u)

    @staticmethod
    def logit(x):
        """Inverse of `sigmoid`.

        Args:
            x: Value in (0, 1).
        """
        return -2.0 * np.sin(np.arcsin(1.0 - 2.0 * x) / 3.0)
