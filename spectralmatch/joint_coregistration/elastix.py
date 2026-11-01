"""Local mutual-information matcher; joint_coregistration owns sampling and solving."""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter


def require_elastix():
    try:
        import itk
    except ImportError as error:
        raise ImportError("Elastix matching requires: pip install 'spectralmatch[elastix]'") from error
    return itk


def match_window(fixed, moving, valid):
    """Return one fixed/moving pixel pair from a local translation, or no match.

    ITK's output-to-input transform maps a fixed pixel to its moving match.
    Coordinates are zero-based pixel centers, as required by joint registration.
    One ITK thread per pair avoids oversubscribing the image process executor.
    """
    empty = np.empty((0, 4), dtype=float)
    valid = valid & np.isfinite(fixed) & np.isfinite(moving)
    if valid.mean() < .5 or min(np.std(fixed[valid]), np.std(moving[valid])) <= 1e-6:
        return empty
    itk = require_elastix()

    def smooth(values):
        # Suppress high-frequency local minima without bleeding nodata into the data.
        weights = gaussian_filter(valid.astype(np.float32), 2)
        values = gaussian_filter(np.where(valid, values, 0).astype(np.float32), 2)
        return itk.image_from_array(np.ascontiguousarray(values / np.maximum(weights, 1e-6)))

    parameters = itk.ParameterObject.New()
    parameter_map = parameters.GetDefaultParameterMap("translation")
    parameter_map["AutomaticTransformInitialization"] = ["false"]
    parameter_map["MaximumNumberOfIterations"] = ["512"]
    parameter_map["MaximumStepLength"] = ["1"]
    parameter_map["NumberOfResolutions"] = ["3"]
    parameter_map["ImageSampler"] = ["Full"]
    parameter_map["NewSamplesEveryIteration"] = ["false"]
    parameters.AddParameterMap(parameter_map)
    mask = itk.image_from_array(np.ascontiguousarray(valid.astype(np.uint8)))
    try:
        _, transform = itk.elastix_registration_method(
            smooth(fixed), smooth(moving), fixed_mask=mask, moving_mask=mask,
            parameter_object=parameters, number_of_threads=1, log_to_console=False,
        )
    except RuntimeError as error:
        warnings.warn(f"Skipping Elastix window: {error}", RuntimeWarning, stacklevel=2)
        return empty
    delta = np.asarray(transform.GetParameterMap(0)["TransformParameters"], dtype=float)
    height, width = valid.shape
    x, y = (width - 1) / 2, (height - 1) / 2
    mx, my = np.array([x, y]) + delta
    if not np.isfinite([mx, my]).all() or not (0 <= mx < width and 0 <= my < height):
        return empty
    if not valid[int(round(y)), int(round(x))] or not valid[min(height-1, int(round(my))), min(width-1, int(round(mx)))]:
        return empty
    return np.array([[x, y, mx, my]])
