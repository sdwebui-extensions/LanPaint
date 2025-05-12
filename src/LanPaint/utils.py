import torch
def epxm1_x(x):
    # Compute the (exp(x) - 1) / x term with a small value to avoid division by zero.
    result = torch.special.expm1(x) / x
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x) < 1e-2
    result = torch.where(mask, 1 + x/2. + x**2 / 6., result)
    return result
def epxm1mx_x2(x):
    # Compute the (exp(x) - 1 - x) / x**2 term with a small value to avoid division by zero.
    result = (torch.special.expm1(x) - x) / x**2
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**2) < 1e-2
    result = torch.where(mask, 1/2. + x/6 + x**2 / 24 + x**3 / 120, result)
    return result

def expm1mxmhx2_x3(x):
    # Compute the (exp(x) - 1 - x - x**2 / 2) / x**3 term with a small value to avoid division by zero.
    result = (torch.special.expm1(x) - x - x**2 / 2) / x**3
    # replace NaN or inf values with 0
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**3) < 1e-2
    result = torch.where(mask, 1/6 + x/24 + x**2 / 120 + x**3 / 720 + x**4 / 5040, result)
    return result

def exp_1mcosh_GD(gamma_t, delta):
    """
    Compute e^(-Γt) * (1 - cosh(Γt√Δ))/ ( (Γt)**2 Δ )
    
    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)
    
    Returns:
    Result of the computation with numerical stability handling
    """
    # Main computation
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta
    numerator_pos =  torch.exp(-gamma_t) - (torch.exp(gamma_t * (sqrt_abs_delta - 1)) + torch.exp(gamma_t * (-sqrt_abs_delta - 1))) / 2
    numerator_neg = torch.exp(-gamma_t) * ( 1 -  torch.cos(gamma_t * sqrt_abs_delta ) )
    numerator = torch.where(is_positive, numerator_pos, numerator_neg)
    result =  numerator / (delta * gamma_t**2 )
    # Handle NaN/inf cases
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    # Handle numerical instability for small delta
    mask = torch.abs(gamma_t_sqrt_delta**2) < 5e-2
    taylor = ( -0.5  - gamma_t**2 / 24 * delta - gamma_t**4 / 720 * delta**2 ) * torch.exp(-gamma_t)
    result = torch.where(mask, taylor, result)
    return result

def exp_sinh_GsqrtD(gamma_t, delta):
    """
    Compute e^(-Γt) * sinh(Γt√Δ) / (Γt√Δ)

    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)

    Returns:
    Result of the computation with numerical stability handling
    """
    # Main computation
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta
    numerator_pos =  (torch.exp(gamma_t * (sqrt_abs_delta - 1)) - torch.exp(gamma_t * (-sqrt_abs_delta - 1))) / 2
    denominator_pos = gamma_t_sqrt_delta
    result_pos = numerator_pos / gamma_t_sqrt_delta
    result_pos = torch.where(torch.isfinite(result_pos), result_pos, torch.zeros_like(result_pos))

    # Taylor expansion for small gamma_t_sqrt_delta
    mask = torch.abs(gamma_t_sqrt_delta) < 1e-2
    taylor = ( 1  + gamma_t**2 / 6 * delta + gamma_t**4 / 120 * delta**2 ) * torch.exp(-gamma_t)
    result_pos = torch.where(mask, taylor, result_pos)

    # Handle negative delta
    result_neg = torch.exp(-gamma_t) * torch.special.sinc(gamma_t_sqrt_delta/torch.pi)
    result = torch.where(is_positive, result_pos, result_neg)
    return result

def exp_cosh(gamma_t, delta):
    """
    Compute e^(-Γt) * cosh(Γt√Δ)

    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)

    Returns:
    Result of the computation with numerical stability handling
    """
    exp_1mcosh_GD_result = exp_1mcosh_GD(gamma_t, delta) # e^(-Γt) * (1 - cosh(Γt√Δ))/ ( (Γt)**2 Δ )
    result = torch.exp(-gamma_t) - gamma_t**2 * delta * exp_1mcosh_GD_result
    return result
def exp_sinh_sqrtD(gamma_t, delta):
    """
    Compute e^(-Γt) * sinh(Γt√Δ) / √Δ
    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)
    Returns:
    Result of the computation with numerical stability handling
    """
    exp_sinh_GsqrtD_result = exp_sinh_GsqrtD(gamma_t, delta) # e^(-Γt) * sinh(Γt√Δ) / (Γt√Δ)
    result = gamma_t * exp_sinh_GsqrtD_result
    return result



def zeta1(gamma_t, delta):
    # Compute hyperbolic terms and exponential
    half_gamma_t = gamma_t / 2
    exp_cosh_term = exp_cosh(half_gamma_t, delta)
    exp_sinh_term = exp_sinh_sqrtD(half_gamma_t, delta)

    
    # Main computation
    numerator = 1 - (exp_cosh_term + exp_sinh_term)
    denominator = gamma_t * (1 - delta) / 4
    result = 1 - numerator / denominator
    
    # Handle numerical instability
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    
    # Taylor expansion for small x (similar to your epxm1Dx approach)
    mask = torch.abs(denominator) < 5e-3
    term1 = epxm1_x(-gamma_t)
    term2 = epxm1mx_x2(-gamma_t)
    term3 = expm1mxmhx2_x3(-gamma_t)
    taylor = term1 + (1/2.+ term1-3*term2)*denominator + (-1/6. + term1/2 - 4 * term2 + 10 * term3) * denominator**2
    result = torch.where(mask, taylor, result)
    
    return result

def exp_cosh_minus_terms(gamma_t, delta):
    """
    Compute E^(-tΓ) * (Cosh[tΓ] - 1 - (Cosh[tΓ√Δ] - 1)/Δ) / (tΓ(1 - Δ))
    
    Parameters:
    gamma_t: Γ*t term (could be a scalar or tensor)
    delta: Δ term (could be a scalar or tensor)
    
    Returns:
    Result of the computation with numerical stability handling
    """
    exp_term = torch.exp(-gamma_t)
    # Compute individual terms
    exp_cosh_term = exp_cosh(gamma_t, gamma_t**0) - exp_term # E^(-tΓ) (Cosh[tΓ] - 1) term
    exp_cosh_delta_term = - gamma_t**2 * exp_1mcosh_GD(gamma_t, delta)  # E^(-tΓ) (Cosh[tΓ√Δ] - 1)/Δ term
    
    #exp_1mcosh_GD e^(-Γt) * (1 - cosh(Γt√Δ))/ ( (Γt)**2 Δ )
    # Main computation
    numerator = exp_cosh_term - exp_cosh_delta_term
    denominator = gamma_t * (1 - delta)
    
    result = numerator / denominator
    
    # Handle numerical instability
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    
    # Taylor expansion for small gamma_t and delta near 1
    mask = (torch.abs(denominator) < 1e-1)
    exp_1mcosh_GD_term = exp_1mcosh_GD(gamma_t, delta**0)
    taylor = (
       gamma_t*exp_1mcosh_GD_term + 0.5 * gamma_t * exp_sinh_GsqrtD(gamma_t, delta**0) 
       - denominator / 4 * ( 0.5 * exp_cosh(gamma_t, delta**0) - 4 * exp_1mcosh_GD_term - 5 /2 * exp_sinh_GsqrtD(gamma_t, delta**0) )
    )
    result = torch.where(mask, taylor, result)
    
    return result


def zeta2(gamma_t, delta):
    half_gamma_t = gamma_t / 2
    return exp_sinh_GsqrtD(half_gamma_t, delta)

def sig11(gamma_t, delta):
    return 1 - torch.exp(-gamma_t) + gamma_t**2 * exp_1mcosh_GD(gamma_t, delta) + exp_sinh_sqrtD(gamma_t, delta)


def Zcoefs(gamma_t, delta):
    Zeta1 = zeta1(gamma_t, delta)
    Zeta2 = zeta2(gamma_t, delta)
    
    sq_total = 1 - Zeta1 + gamma_t * (delta - 1) * (Zeta1 - 1)**2 / 8
    amplitude = torch.sqrt(sq_total)
    Zcoef1 = ( gamma_t**0.5 * Zeta2 / 2 **0.5 ) / amplitude
    Zcoef2 = Zcoef1 * gamma_t *( - 2 * exp_1mcosh_GD(gamma_t, delta)  / sig11(gamma_t, delta)  ) ** 0.5 
    #cterm = exp_cosh_minus_terms(gamma_t, delta)
    #sterm = exp_sinh_sqrtD(gamma_t, delta**0) + exp_sinh_sqrtD(gamma_t, delta)
    #Zcoef3 = 2 * torch.sqrt(  cterm / ( gamma_t * (1 - delta) * cterm + sterm ) )
    Zcoef3 = torch.sqrt( torch.maximum(1 - Zcoef1**2 - Zcoef2**2, sq_total.new_zeros(sq_total.shape)) )

    return Zcoef1 * amplitude, Zcoef2 * amplitude, Zcoef3 * amplitude, amplitude

def Zcoefs_asymp(gamma_t, delta):
    A_t = gamma_t * (1 - delta) 
    return epxm1_x(- 2 * A_t)