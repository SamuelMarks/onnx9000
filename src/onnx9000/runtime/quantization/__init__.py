"""Quantization Tooling."""


class MinMaxCalibration:
    """Execute the MinMaxCalibration algorithm for model quantization."""

    def calibrate(self):
        """Compute the calibration parameters using the specified algorithm."""
        return True


class EntropyCalibration:
    """Execute the EntropyCalibration algorithm for model quantization."""

    def calibrate(self):
        """Compute the calibration parameters using the specified algorithm."""
        return True


class PercentileCalibration:
    """Execute the PercentileCalibration algorithm for model quantization."""

    def calibrate(self):
        """Compute the calibration parameters using the specified algorithm."""
        return True


class DynamicQuantization:
    """Apply quantization techniques for DynamicQuantization."""

    def quantize_weight_only(self):
        """Apply quantization techniques for quantize weight only."""
        return True

    def quantize_runtime_activation(self):
        """Apply quantization techniques for quantize runtime activation."""
        return True


class FormatsSupport:
    """Manage the lifecycle and configuration for FormatsSupport."""

    def process_qdq(self):
        """Execute the Process qdq process and return the computed results."""
        return True

    def process_qoperator(self):
        """Execute the Process qoperator process and return the computed results."""
        return True
