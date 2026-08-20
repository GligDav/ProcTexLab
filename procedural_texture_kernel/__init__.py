"""Raster-to-procedural-texture computational kernel."""
from .api import (DEFAULT_DETAIL_COMPONENT_FAMILIES, FitConfig, FitResult,
                  SUPPORTED_COMPONENT_FAMILIES, TextureFitter)
from .components import *
from .io import load_image, normalize_image
from .model import ProceduralTextureModel
from .shader_graph import ShaderGraph, ShaderGraphComponent, ShaderNode
from .decomposition import ImageDecomposition, LaplacianPyramid, create_decomposition
from .texture_loss import TextureLossWeights, calculate_texture_loss
from .spectral_diagnostics import RadialPowerSpectrum, compare_spectra, radial_power_spectrum
from .measurement_diagnostics import compare_measurements
from .weight_estimator import (BandFeatureExtractor, BandFeatures, LossWeights,
                               WeightEstimator, WeightEstimatorConfig,
                               WeightEstimatorResult, WeightMappingConfig)
__all__ = ["FitConfig", "FitResult", "TextureFitter", "ProceduralTextureModel",
           "ProceduralComponent", "SinusoidComponent", "GaborComponent",
           "GaussianRBFComponent", "PerlinNoiseComponent", "ThresholdedNoiseComponent",
           "MaskedNoiseComponent", "WaveletComponent",
           "ShaderNode", "ShaderGraph", "ShaderGraphComponent",
           "VoronoiNoiseComponent", "FractalBrownianMotionComponent",
           "RidgedMultifractalComponent", "TurbulenceNoiseComponent",
           "DomainWarpedNoiseComponent", "WarpedRidgedMultifractalComponent",
           "AnisotropicGaussianComponent", "LineComponent",
           "StepEdgeComponent", "DifferenceOfGaussiansComponent", "PolynomialTrendComponent",
           "RadialWaveComponent", "SpiralWaveComponent", "SparseImpulseComponent",
           "BinaryPrimitiveComponent",
           "load_image", "normalize_image"]
__all__ += ["TextureLossWeights", "calculate_texture_loss"]
__all__ += ["RadialPowerSpectrum", "radial_power_spectrum", "compare_spectra"]
__all__ += ["compare_measurements"]
__all__ += ["SUPPORTED_COMPONENT_FAMILIES"]
__all__ += ["DEFAULT_DETAIL_COMPONENT_FAMILIES"]
__all__ += ["ImageDecomposition", "LaplacianPyramid", "create_decomposition"]
__all__ += ["BandFeatureExtractor", "BandFeatures", "LossWeights", "WeightEstimator",
            "WeightEstimatorConfig", "WeightEstimatorResult", "WeightMappingConfig"]
