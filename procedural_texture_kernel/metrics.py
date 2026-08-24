"""Numerical reconstruction metrics."""
import numpy as np

def calculate_metrics(
    target: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """Compare two same-shaped images with scalar error metrics.

    Args:
        target: Non-empty reference image.
        reconstruction: Reconstructed image with the same shape as ``target``.

    Returns:
        MSE, RMSE, MAE, PSNR, normalized RMSE, and correlation by name.
    """
    a, b = np.asarray(target, float), np.asarray(reconstruction, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("metric inputs must be non-empty and have matching shapes")
    d = a - b
    mse = float(np.mean(d * d)); rmse = float(np.sqrt(mse)); mae = float(np.mean(np.abs(d)))
    dynamic_range = float(np.ptp(a))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    nrmse = 0.0 if rmse == 0 else (float("inf") if dynamic_range == 0 else rmse / dynamic_range)
    if np.std(a) == 0 or np.std(b) == 0:
        correlation = 1.0 if np.array_equal(a, b) else 0.0
    else:
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return {"mse": mse, "rmse": rmse, "mae": mae, "psnr": psnr,
            "normalized_rmse": nrmse, "correlation": correlation}
