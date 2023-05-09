import pytensor.tensor as pt

def pt_interp(x, xs, ys):
    """Linear interpolation: ``f(x; xs, ys)``"""
    x = pt.as_tensor(x)
    xs = pt.as_tensor(xs)
    ys = pt.as_tensor(ys)

    n = xs.shape[0]
    
    inds = pt.searchsorted(xs, x)
    inds = pt.where(inds <= 0, 1, inds)
    inds = pt.where(inds > n-1, n-1, inds)
    
    r = (x - xs[inds-1]) / (xs[inds] - xs[inds-1])

    return r*ys[inds] + (1-r)*ys[inds-1]

def pt_cumtrapz(ys, xs):
    """Cumulative trapezoidal integral of the function defined by values `ys` at points `xs`."""
    xs = pt.as_tensor(xs)
    ys = pt.as_tensor(ys)

    return pt.cumsum(pt.concatenate([pt.as_tensor([0.0]), 0.5*(ys[1:] + ys[:-1])*pt.diff(xs)]))