"""Literal schema-v1 analytic formulas (frozen from the fitter kernel)."""
from __future__ import annotations

import math

TAU = 2.0 * math.pi


def _rotated(n, u, v, c):
    du, dv = n.sub(u, c["center_u"]), n.sub(v, c["center_v"])
    co, si = math.cos(c.get("orientation", 0.0)), math.sin(c.get("orientation", 0.0))
    return n.add(n.mul(co, du), n.mul(si, dv)), n.add(n.mul(-si, du), n.mul(co, dv))


def _amp(n, basis, c): return n.mul(c["amplitude"], basis)


def sinusoid(n, u, v, c):
    phase = n.add(n.mul(TAU, n.add(n.mul(c["frequency_u"], u), n.mul(c["frequency_v"], v))), c["phase"])
    return _amp(n, n.math("COSINE", phase), c)


def spectral(n, u, v, c):
    modes = []
    for fu, fv, weight, phase in zip(c["frequencies_u"], c["frequencies_v"], c["weights"], c["phases"]):
        angle = n.add(n.mul(TAU, n.add(n.mul(fu, u), n.mul(fv, v))), phase)
        modes.append(n.mul(weight, n.math("COSINE", angle)))
    rms = max(math.sqrt(0.5 * sum(w * w for w in c["weights"])), 1e-12)
    return _amp(n, n.div(n.balanced_add(modes), rms), c)


def _ellipse(n, u, v, c):
    x, y = _rotated(n, u, v, c)
    q = n.add(n.power(n.div(x, c["sigma_u"]), 2), n.power(n.div(y, c["sigma_v"]), 2))
    return x, n.math("EXPONENT", n.mul(-0.5, q))


def gabor(n, u, v, c):
    x, envelope = _ellipse(n, u, v, c)
    carrier = n.math("COSINE", n.add(n.mul(TAU * c["frequency"], x), c["phase"]))
    return _amp(n, n.mul(envelope, carrier), c)


def anisotropic(n, u, v, c): return _amp(n, _ellipse(n, u, v, c)[1], c)


def gaussian(n, u, v, c):
    du, dv = n.sub(u, c["center_u"]), n.sub(v, c["center_v"])
    r2 = n.add(n.power(du, 2), n.power(dv, 2))
    return _amp(n, n.math("EXPONENT", n.div(n.mul(-0.5, r2), c["sigma"] ** 2)), c)


def wavelet(n, u, v, c):
    x, y = _rotated(n, u, v, c)
    r2 = n.add(n.power(n.div(x, c["scale_u"]), 2), n.power(n.div(y, c["scale_v"]), 2))
    return _amp(n, n.mul(n.sub(1, r2), n.math("EXPONENT", n.mul(-0.5, r2))), c)


def line(n, u, v, c):
    x, y = _rotated(n, u, v, c)
    d = n.math("MAXIMUM", n.sub(n.math("ABSOLUTE", y), c["width"] / 2), n.sub(n.math("ABSOLUTE", x), c["length"] / 2))
    # Logistic written as 1/(1+exp(clamp(d/s,-60,60))). Blender Clamp is [0,1],
    # so use explicit min/max for the kernel's symmetric clamp.
    z = n.math("MINIMUM", n.math("MAXIMUM", n.div(d, max(c["softness"], 1e-12)), -60), 60)
    return _amp(n, n.div(1, n.add(1, n.math("EXPONENT", z))), c)


def step(n, u, v, c):
    x, _ = _rotated(n, u, v, c)
    basis = n.math("GREATER_THAN", x, -1e-12) if c["softness"] <= 0 else n.math("TANH", n.div(x, c["softness"]))
    if c["softness"] <= 0: basis = n.sub(n.mul(2, basis), 1)
    return _amp(n, basis, c)


def dog_log(n, u, v, c):
    du, dv = n.sub(u, c["center_u"]), n.sub(v, c["center_v"])
    r2 = n.add(n.power(du, 2), n.power(dv, 2)); q = n.div(r2, c["sigma"] ** 2)
    inner = n.math("EXPONENT", n.mul(-0.5, q))
    if c.get("mode", "dog") == "log": basis = n.mul(n.sub(1, n.mul(0.5, q)), inner)
    else:
        ratio = max(c["ratio"], 1.000001)
        outer = n.div(n.math("EXPONENT", n.div(n.mul(-0.5, r2), (ratio * c["sigma"]) ** 2)), ratio ** 2)
        basis = n.sub(inner, outer)
    return _amp(n, basis, c)


def polynomial(n, u, v, c):
    x, y = n.sub(u, .5), n.sub(v, .5)
    terms = [n.mul(c["linear_u"], x), n.mul(c["linear_v"], y), n.mul(c["quadratic_u"], n.mul(x, x)),
             n.mul(c["cross_uv"], n.mul(x, y)), n.mul(c["quadratic_v"], n.mul(y, y))]
    return _amp(n, n.balanced_add(terms), c)


def _radial_parts(n, u, v, c):
    du, dv = n.sub(u, c["center_u"]), n.sub(v, c["center_v"])
    return du, dv, n.math("SQRT", n.add(n.mul(du, du), n.mul(dv, dv)))


def radial(n, u, v, c):
    _, _, r = _radial_parts(n, u, v, c)
    wave = n.math("COSINE", n.add(n.mul(TAU * c["frequency"], r), c["phase"]))
    return _amp(n, n.mul(wave, n.math("EXPONENT", n.mul(-c["decay"], r))), c)


def spiral(n, u, v, c):
    du, dv, r = _radial_parts(n, u, v, c)
    theta = n.math("ARCTAN2", dv, du)
    phase = n.balanced_add([n.mul(TAU * c["frequency"], r), n.mul(c["arms"], theta), n.value(c["phase"])])
    return _amp(n, n.mul(n.math("COSINE", phase), n.math("EXPONENT", n.mul(-c["decay"], r))), c)


def binary(n, u, v, c):
    x, y = _rotated(n, u, v, c); shape = c["shape"]
    if shape == "disk": basis = n.math("LESS_THAN", n.add(n.power(n.div(x, c["size_u"]), 2), n.power(n.div(y, c["size_v"]), 2)), 1.0000001)
    elif shape == "box": basis = n.mul(n.math("LESS_THAN", n.math("ABSOLUTE", x), c["size_u"]), n.math("LESS_THAN", n.math("ABSOLUTE", y), c["size_v"]))
    elif shape == "ring":
        radius = n.math("SQRT", n.add(n.power(n.div(x, c["size_u"]), 2), n.power(n.div(y, c["size_v"]), 2)))
        basis = n.math("LESS_THAN", n.math("ABSOLUTE", n.sub(radius, 1)), c["thickness"] / max(c["size_u"], c["size_v"]))
    else:
        cells = n.add(n.math("FLOOR", n.div(x, c["size_u"])), n.math("FLOOR", n.div(y, c["size_v"])))
        basis = n.math("LESS_THAN", n.math("MODULO", cells, 2), 1)
    return _amp(n, basis, c)


def constant(n, u, v, c): return n.value(c["amplitude"] * c["value"], "Constant component")


ANALYTIC_BUILDERS = {
    "sinusoid": sinusoid, "spectral_noise": spectral, "gabor": gabor,
    "gaussian_rbf": gaussian, "wavelet": wavelet, "anisotropic_gaussian": anisotropic,
    "line": line, "step_edge": step, "dog_log": dog_log,
    "polynomial_trend": polynomial, "radial_wave": radial, "spiral_wave": spiral,
    "binary_primitive": binary, "simple_constant": constant,
}
