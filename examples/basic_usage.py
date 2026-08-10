"""Fit a known target using only the public API."""
from pathlib import Path
import numpy as np
from PIL import Image
from procedural_texture_kernel import FitConfig, ProceduralTextureModel, SinusoidComponent, TextureFitter

def main():
    truth = ProceduralTextureModel(bias=.48, components=[
        SinusoidComponent(amplitude=.24, frequency_u=3, frequency_v=1, phase=.4),
        SinusoidComponent(amplitude=.12, frequency_u=-1, frequency_v=6, phase=-.8)])
    target = truth.evaluate(96, 64)
    result = TextureFitter(FitConfig(max_components=4, fitting_resolution=96)).fit(target)
    print("Components:", len(result.model.components)); print("Metrics:", result.metrics)
    out = Path("example_output"); out.mkdir(exist_ok=True)
    for name, data in (("target", target), ("reconstruction", result.reconstruction),
                       ("residual", .5 + result.residual)):
        Image.fromarray(np.uint8(np.clip(data, 0, 1) * 255)).save(out / f"{name}.png")
    result.save_json(out / "model.json")
if __name__ == "__main__": main()
