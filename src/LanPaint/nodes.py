from contextlib import contextmanager
from inspect import cleandoc
import inspect
# import nodes.py
import comfy
import nodes
import latent_preview
from functools import partial
from comfy.utils import repeat_to_batch_size
from comfy.samplers import *
from comfy.model_base import ModelType
from .utils import *
# Monkey patch comfy.samplers module by importing with absolute package path
#exec(inspect.getsource(comfy.samplers).replace("from .", "from comfy."))

def reshape_mask(input_mask, output_shape):
    dims = len(output_shape) - 2


    scale_mode = "nearest-exact"
    mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
    if mask.shape[1] < output_shape[1]:
        mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:,:output_shape[1]]
    mask = repeat_to_batch_size(mask, output_shape[0])
    return mask
def prepare_mask(noise_mask, shape, device):
    return reshape_mask(noise_mask, shape).to(device)
def sampling_function_LanPaint(model, x, timestep, uncond, cond, cond_scale, cond_scale_BIG, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out  = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_), cfg_function(model, out[0], out[1], cond_scale_BIG, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)


class CFGGuider_LanPaint:
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        print("CFGGuider outer_sample")
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function_LanPaint(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, self.cfg_BIG, model_options=model_options, seed=seed)

#CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample
#CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise

class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
        self.model_sigmas = torch.cat( (torch.tensor([0.], device = sigmas.device) , torch.tensor( self.inner_model.model_patcher.get_model_object("model_sampling").sigmas, device = sigmas.device) ) )
        self.model_sigmas = torch.tensor( self.model_sigmas, dtype = self.sigmas.dtype )
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None,**kwargs):
        ### For 1.5 and XL model
        # x is x_t in the notation of variance exploding diffusion model, x_t = x_0 + sigma * noise
        # sigma is the noise level
        ### For flux model 
        # x is rectified flow x_t = sigma * noise + (1.0 - sigma) * x_0

        IS_FLUX = self.inner_model.inner_model.model_type == ModelType.FLUX
        IS_FLOW = self.inner_model.inner_model.model_type == ModelType.FLOW

        # unify the notations into variance exploding diffusion model
        if IS_FLUX or IS_FLOW:
            VE_Sigma = sigma / ( torch.maximum( 1 - sigma , sigma*0 + 5e-2  ))
            self.VE_Sigmas = self.sigmas / ( torch.maximum( 1 - self.sigmas , self.sigmas*0 + 5e-2  ))
        else:
            VE_Sigma = sigma 
            self.VE_Sigmas = self.sigmas

        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})

            denoise_mask = (denoise_mask > 0.5).float()

            latent_mask = 1 - denoise_mask

            abt = 1/( 1+VE_Sigma**2 )

            current_times = (VE_Sigma, abt)

            out = self.LanPaint(x, sigma, latent_mask, current_times, model_options, seed, IS_FLUX, IS_FLOW, denoise_mask)
        else:
            out, _ = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        
        # Add TAESD preview support - directly use the latent_preview module
        current_step = model_options.get("i", kwargs.get("i", 0))
        total_steps = model_options.get("total_steps", 0)

        # Only show preview every few steps to improve performance
        if current_step % 2 == 0:
            # Directly call the preview callback if it exists
            callback = model_options.get("callback", None)
            if callback is not None:
                callback({"i": current_step, "denoised": out, "x": x})
    
        return out
    def LanPaint(self, x, sigma, latent_mask, current_times, model_options, seed, IS_FLUX, IS_FLOW, denoise_mask):
        VE_Sigma, abt = current_times
        sigma_ind = torch.argmin(torch.abs(self.VE_Sigmas - torch.mean( VE_Sigma )))
        VE_Sigma_next = self.VE_Sigmas[ sigma_ind + 1 ] * VE_Sigma**0

        
        #step_size = self.step_size * (1 - abt) ** b * abt ** a / ( ((a/(a+b))**a*(b/(a+b))**b) )

        if self.step_time_schedule == "dual_shrink":
            step_size = self.step_size * (1 - abt) ** 0.5 * abt ** 0.5
        elif self.step_time_schedule == "follow_sampler":
            tt = torch.log(1+VE_Sigma**2)
            tt_next = torch.log(1+VE_Sigma_next**2)
            tt_first = torch.log(1+self.VE_Sigmas[0]**2)
            tt_second = torch.log(1+self.VE_Sigmas[1]**2)
            #print("max t step",tt_first - tt_second)
            step_size = (tt - tt_next) / (tt_first - tt_second) * self.step_size * abt ** 0
        else:
            abt_end = 1/( 1+self.end_sigma**2 ) 
            step_size = self.step_size * (1 -  torch.minimum(abt/abt_end, abt**0) ) ** 0.5



        step_size = step_size[:, None, None, None]
        # self.inner_model.inner_model.scale_latent_inpaint returns variance exploding x_t values
        x = x * (1 - latent_mask) +  self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image)* latent_mask
        
        if IS_FLUX or IS_FLOW:
            x_t = x * ( 1 + VE_Sigma[:, None,None,None])
        else:
            x_t = x #/ ( 1+sigma**2 )**0.5 # switch to variance perserving x_t values

        ############ LanPaint Iterations Start ###############
        # after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        args = None
        for i in range(self.n_steps):
            if torch.mean(VE_Sigma) > self.start_sigma or torch.mean(VE_Sigma) < self.end_sigma:
                
                break

            score_func = partial( self.score_model, y = self.latent_image, mask = latent_mask, abt = abt[:, None,None,None], sigma = VE_Sigma[:, None,None,None], model_options = model_options, seed = seed )
            if self.step_size_schedule == "linear":
                step_size_i = step_size * (1 - i/(self.n_steps) )
            else:
                step_size_i = step_size 
            x_t, args = self.langevin_dynamics(x_t, score_func , latent_mask, step_size_i , current_times, sigma_x = self.sigma_x(abt)[:, None,None,None], sigma_y = self.sigma_y(abt)[:, None,None,None], args = args)  
        if IS_FLUX or IS_FLOW:
            x = x_t / ( 1 + VE_Sigma[:, None,None,None] )
        else:
            x = x_t #/ ( 1+sigma**2 )**0.5 # switch to variance perserving x_t values
        ############ LanPaint Iterations End ###############
        # out is x_0
        out, _ = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        out = out * denoise_mask + self.latent_image * latent_mask
        return out
    def truncate_times(self, current_times, step_size):
        sigma, abt = current_times
        tt = torch.log(1+sigma**2)
        tt_mid = torch.max( tt - step_size, tt*0 )
        step_size = tt - tt_mid
        return step_size
    def mid_times(self, current_times, step_size):
        sigma, abt = current_times
        tt = torch.log(1+sigma**2)
        tt_mid = torch.max( tt - step_size, tt*0 )
        sigma_mid = (torch.exp(tt_mid) - 1) ** 0.5
        abt_mid = 1/(1+sigma_mid**2)
        
        if torch.mean(sigma) >= self.LanPaint_Cap_Sigma:
            return tt - tt_mid, sigma_mid, abt_mid
        else:
            sigma_ind = torch.argmin(torch.abs(self.VE_Sigmas - torch.mean( sigma )))
            VE_Sigma_next = self.VE_Sigmas[ sigma_ind + 1 ] * sigma**0
            abt_next = 1/( 1+VE_Sigma_next**2 )
            tt_next = torch.log(1+VE_Sigma_next**2)
            return tt - tt_next, VE_Sigma_next, abt_next
    def score_model(self, x_t, y, mask, abt, sigma, model_options, seed):
        
        lamb = self.chara_lamb

        IS_FLUX = self.inner_model.inner_model.model_type == ModelType.FLUX
        IS_FLOW = self.inner_model.inner_model.model_type == ModelType.FLOW
        if IS_FLUX or IS_FLOW:
            x_0, x_0_BIG = self.inner_model(x_t / ( 1 + sigma ), sigma[:, 0,0,0] / ( 1 + sigma[:, 0,0,0] ), model_options=model_options, seed=seed)
        else:
            x_0, x_0_BIG = self.inner_model(x_t, sigma[:, 0,0,0], model_options=model_options, seed=seed)

        e_t = x_t / ((1 - abt) ** 0.5 * (1 + sigma**2) ** 0.5 )- (abt ** 0.5  / (1 - abt) ** 0.5) * x_0
        e_t_BIG = x_t / ((1 - abt) ** 0.5 * (1 + sigma**2) ** 0.5 )- (abt ** 0.5  / (1 - abt) ** 0.5) * x_0_BIG
        
        score_x = -e_t
        score_y = - (1 + lamb) * ( x_t/ ((1 + sigma**2) ** 0.5 *(1 - abt)**0.5) - abt**0.5 /(1 - abt)**0.5 * y )  + lamb *  e_t_BIG
        return score_x * (1 - mask) + score_y * mask
    def sigma_x(self, abt):
        # the time scale for the x_t update
        return abt**0
    def sigma_y(self, abt):
        # the time scale for the y_t update
        if self.beta_scale == "shrink":
            beta = self.chara_beta * (1-abt)**0.5
        elif self.beta_scale == "dual_shrink":
            beta = self.chara_beta * (1-abt)**0.5 * abt ** 0.5
        elif self.beta_scale == "back_shrink":
            beta = self.chara_beta * abt ** 0.5
        else:
            beta = self.chara_beta * abt ** 0
        return beta
    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):

        # prepare the step size and time parameters
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            step_sizes = self.prepare_step_size(current_times, step_size, sigma_x, sigma_y)
            sigma, dtx, dty, Gamma_hat_x, Gamma_hat_y, sigma_mid_x, sigma_mid_y, A_t_x, A_t_y = step_sizes

        if torch.mean(sigma_mid_x) >= torch.mean(sigma) or torch.mean(sigma_mid_y) >= torch.mean(sigma):
            return x_t, args
        # -------------------------------------------------------------------------
        # A: Update epsilon (score estimate and noise initialization)
        # -------------------------------------------------------------------------
        eps_momentum, eps_model, Z = self.eps_evalutation(x_t, score, args)
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            x_t, eps_momentum, Z = self.Langevin_step(x_t, eps_model, eps_momentum, Z, mask, step_sizes)
        return x_t, (eps_momentum, Z)
    def Langevin_step(self, x_t, eps_model, eps_momentum, Z, mask, step_sizes):
        sigma, dtx, dty, Gamma_hat_x, Gamma_hat_y, sigma_mid_x, sigma_mid_y, A_t_x, A_t_y = step_sizes
        # -------------------------------------------------------------------------
        # B: Update epsilon and noise Z with momentum
        # -------------------------------------------------------------------------
        # Compute the weighted combination term for epsilon mean update:
        # term = (2/Γ_hat)*(1-exp(-0.5*Γ_hat))
        eps_momentum, eps_denoise = self.eps_with_momentum(eps_momentum, eps_model, Gamma_hat_x, Gamma_hat_y, A_t_x, A_t_y, mask)
        # tamed eps for more stable updates
        #eps_denoise = self.tamed_eps(eps_denoise, mask, sigma, dtx, dty)
        Z_comb, Z_next = self.noise_with_momentum(Z, Gamma_hat_x, Gamma_hat_y, A_t_x, A_t_y, mask, x_t)
        # -------------------------------------------------------------------------
        # C: Langevin Dynamics splitted into denoise-addnoise steps with Euler Scheme
        # -------------------------------------------------------------------------
        # Transform x to z using z = x * sqrt(1+sigma^2). Here we have already set x to z to avoid floating point stability issue.
        z_t = x_t #* (1 + sigma**2) ** 0.5
        # Denoise
        z_mid_x = z_t + eps_denoise * (sigma_mid_x - sigma)
        z_mid_y = z_t - eps_denoise * (1+sigma**2) / sigma * dty / 2   
        z_mid = z_mid_x * (1 - mask) + z_mid_y * mask
        # Compute the change in noise level sigma (dsigma = sqrt(sigma^2 - sigma_mid^2)).
        dsigma_x = sigma * torch.sqrt(1 - (sigma_mid_x / sigma) ** 2)
        dsigma_y = torch.sqrt( (1 + sigma**2) * dty ) 
        dsigma = dsigma_x * (1 - mask) + dsigma_y * mask
        # Add noise
        z_final = z_mid + Z_comb * dsigma
        # Transform back to x-space: x = z / sqrt(1+sigma^2)
        x_t = z_final #/ (1 + sigma**2) ** 0.5
        return x_t, eps_momentum, Z_next
    def prepare_step_size(self, current_times, step_size, sigma_x, sigma_y):
        # -------------------------------------------------------------------------
        # Unpack current times parameters (sigma and abt)
        sigma, abt = current_times
        sigma = sigma[:, None,None,None]
        abt = abt[:, None,None,None]
        # Compute time step (dtx, dty) for x and y branches.
        dtx = 2 * step_size * sigma_x
        dty = 2 * step_size * sigma_y
        
        # Get mid time parameters (sigma_mid and abt_mid) for each branch.
        dtx, sigma_mid_x, abt_mid_x = self.mid_times((sigma, abt), dtx)
        dty, sigma_mid_y, abt_mid_y = self.mid_times((sigma, abt), dty)

        #ref_dt = 0.1 * (1 - abt) ** b * abt ** a / ( ((a/(a+b))**a*(b/(a+b))**b) )
        abt_end = 1/( 1+self.end_sigma**2 ) 
        ref_dt = 0.1 * (1 -  torch.minimum(abt/abt_end, abt**0) ) ** 0.5
        ref_dtx = self.truncate_times(current_times, torch.squeeze(2 * sigma_x * ref_dt))
        ref_dty = torch.squeeze(2 * sigma_y * ref_dt)
        # -------------------------------------------------------------------------
        # Define friction parameter Gamma_hat for each branch.
        # Using dtx**0 provides a tensor of the proper device/dtype.

        Gamma_hat_x = self.friction * self.step_size * sigma_x / 0.1 * sigma**0
        Gamma_hat_y = self.friction * self.step_size * sigma_y / 0.1 * sigma**0
        print("Gamma_hat_x", torch.mean(Gamma_hat_x).item(), "Gamma_hat_y", torch.mean(Gamma_hat_y).item())
        # adjust dt to match denoise-addnoise steps sizes
        Gamma_hat_x /= 2.
        Gamma_hat_y /= 2.

        A_t_x = 0 * dtx / 2
        A_t_y =  (1+self.chara_lamb) / ( 1 - abt ) * dty / 2


        return sigma, dtx, dty, Gamma_hat_x, Gamma_hat_y, sigma_mid_x, sigma_mid_y, A_t_x, A_t_y
    def eps_evalutation(self, x_t, score, args):
        score_model = score(x_t)
        eps_model = -score_model 
        # Initialize epsilon and Z if not provided in args.
        if args is None:
            eps = eps_model
            Z = torch.randn_like(x_t)
        else:
            eps, Z = args
        return eps, eps_model, Z
    def tamed_eps(self, eps_denoise, mask, sigma, dtx, dty):
        # tamed
        eps_model_x = eps_denoise* (1 - mask)
        eps_model_x = eps_model_x* ((torch.sum(1 - mask, dim = (1,2,3), keepdim = True)/torch.sum(eps_model_x**2, dim = (1,2,3), keepdim = True)) **0.5) ** torch.minimum(self.tamed*(dtx),sigma**0)#/( 1 + self.tamed*(sigma - sigma_mid_x) * (torch.sum(eps_model_x**2)/torch.sum((1 - mask)))**0.5 )
        eps_model_y = eps_denoise* mask
        eps_model_y = eps_model_y* ((torch.sum(mask, dim = (1,2,3), keepdim = True)/torch.sum(eps_model_y**2, dim = (1,2,3), keepdim = True)) **0.5) ** torch.minimum(self.tamed*(dty),sigma**0)#/( 1 + self.tamed*(sigma - sigma_mid_y) * (torch.sum(eps_model_y**2)/torch.sum(mask))**0.5 )
        eps_denoise = eps_model_x * (1 - mask) + eps_model_y * mask
        return eps_denoise
    def eps_with_momentum(self, eps, eps_model, Gamma_hat_x, Gamma_hat_y, A_t_x, A_t_y, mask):

        # Compute the coefficients for the momentum update.
        Delta_x = 1 - 4 * A_t_x / Gamma_hat_x
        Delta_y = 1 - 4 * A_t_y / Gamma_hat_y
        zeta_1_x = zeta1( Gamma_hat_x, Delta_x) 
        zeta_1_y = zeta1( Gamma_hat_y, Delta_y)
        zeta_2_x = zeta2( Gamma_hat_x, Delta_x)
        zeta_2_y = zeta2( Gamma_hat_y, Delta_y)
        e_x = 1 - Gamma_hat_x * zeta_2_x
        e_y = 1 - Gamma_hat_y * zeta_2_y

        Gamma_hat_asymp_x, Gamma_hat_asymp_y = 1e4 * Gamma_hat_x, 1e4 * Gamma_hat_y
        Delta_asymp_x, Delta_asymp_y = 1 - 4 * A_t_x / Gamma_hat_asymp_x , 1 - 4 * A_t_y / Gamma_hat_asymp_y

        zeta_1_asymp_x, zeta_2_asymp_x = zeta1( Gamma_hat_asymp_x, Delta_asymp_x), zeta2( Gamma_hat_asymp_x, Delta_asymp_x)
        zeta_1_asymp_y, zeta_2_asymp_y = zeta1( Gamma_hat_asymp_y, Delta_asymp_y), zeta2( Gamma_hat_asymp_y, Delta_asymp_y)


        eps_bar_x = self.alpha * ( zeta_2_x * eps + (1-zeta_1_x) * eps_model) + (1 - self.alpha) * ( zeta_2_asymp_x * eps + (1-zeta_1_asymp_x) * eps_model)
        eps_bar_y = self.alpha * ( zeta_2_y * eps + (1-zeta_1_y) * eps_model) + (1 - self.alpha) * ( zeta_2_asymp_y * eps + (1-zeta_1_asymp_y) * eps_model)
         # Form the denoised epsilon using self.alpha (assumed to be 1/Ψ)
        eps_denoise = eps_bar_x * (1 - mask) + eps_bar_y * mask 

       

        # Update the mean epsilon for the next step:
        eps_x = eps * (e_x - A_t_x * ( 1 - zeta_1_x ) )+ eps_model * (1 - e_x)
        eps_y = eps * (e_y - A_t_y * ( 1 - zeta_1_y ) )+ eps_model * (1 - e_y) 
        eps = eps_x * (1 - mask) + eps_y * mask
        return eps, eps_denoise
    def noise_with_momentum(self, Z, Gamma_hat_x, Gamma_hat_y, A_t_x, A_t_y, mask, x_t):

        Delta_x = 1 - 4 * A_t_x / Gamma_hat_x
        Delta_y = 1 - 4 * A_t_y / Gamma_hat_y
        zeta_1_x = zeta1( Gamma_hat_x, Delta_x) 
        zeta_1_y = zeta1( Gamma_hat_y, Delta_y)
        zeta_2_x = zeta2( Gamma_hat_x, Delta_x)
        zeta_2_y = zeta2( Gamma_hat_y, Delta_y)
        e_x = 1 - Gamma_hat_x * zeta_2_x
        e_y = 1 - Gamma_hat_y * zeta_2_y

        Sig11_x = sig11(Gamma_hat_x, Delta_x)
        Sig11_y = sig11(Gamma_hat_y, Delta_y)

        Zcoefs_asymp_x = Zcoefs_asymp( Gamma_hat_x, Delta_x) 
        Zcoefs_asymp_y = Zcoefs_asymp( Gamma_hat_y, Delta_y)


        # Generate auxiliary noise terms.
        Z_q     = torch.randn_like(x_t)
        Z_q_avg = torch.randn_like(x_t)
        Z_z     = torch.randn_like(x_t)

        # Update Z for each branch:
        Z_x = (e_x - A_t_x * ( 1 - zeta_1_x ) ) * Z + (Sig11_x) ** 0.5 * Z_q
        Z_y = (e_y - A_t_y * ( 1 - zeta_1_y ) ) * Z + (Sig11_y) ** 0.5 * Z_q
        Z_next = Z_x * (1 - mask) + Z_y * mask



        def noise_generation(Gamma_hat, Delta, Z, Z_q, Z_q_avg):
            term1, term2, term3, amplitude = Zcoefs( Gamma_hat, Delta) 
            terms = torch.stack([term1, term2, term3], dim=-1)
            #terms = torch.nn.functional.normalize(terms, dim=-1)

            Zs = torch.stack([Z, Z_q, Z_q_avg], dim=-1)
            return torch.sum( terms* Zs, dim=-1 )
        # Compute the combined noise update following the scheme:
        Z_comb_x = noise_generation(Gamma_hat_x, Delta_x, Z, Z_q, Z_q_avg)
        Z_comb_y = noise_generation(Gamma_hat_y, Delta_y, Z, Z_q, Z_q_avg)

        Z_comb = Z_comb_x * (1 - mask) + Z_comb_y * mask

        Z_asymp = Z_z * Zcoefs_asymp_x ** 0.5 * (1 - mask) + Z_z * Zcoefs_asymp_y ** 0.5 * mask


        # Combine with an additional noise term using self.alpha.
        Z_comb = self.alpha ** 0.5 * Z_comb + (1 - self.alpha) ** 0.5 * Z_asymp
        return Z_comb, Z_next

# Custom sampler class extending ComfyUI's KSAMPLER for LanPaint
class KSAMPLER(comfy.samplers.KSAMPLER):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        #noise here is a randn noise from comfy.sample.prepare_noise
        #latent_image is the latent image as input of the KSampler node. For inpainting, it is the masked latent image. Otherwise it is zero tensor.
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        IS_FLUX = model_wrap.inner_model.model_type == ModelType.FLUX

        # unify the notations into variance exploding diffusion model
        if IS_FLUX:
            model_wrap.cfg_BIG = 1.0
        else:
            model_wrap.cfg_BIG = model_wrap.model_patcher.LanPaint_cfg_BIG

        model_k.step_size = model_wrap.model_patcher.LanPaint_StepSize
        model_k.chara_lamb = model_wrap.model_patcher.LanPaint_Lambda
        model_k.chara_beta = model_wrap.model_patcher.LanPaint_Beta
        model_k.n_steps = model_wrap.model_patcher.LanPaint_NumSteps
        model_k.friction = model_wrap.model_patcher.LanPaint_Friction
        model_k.alpha = model_wrap.model_patcher.LanPaint_Alpha
        model_k.tamed = model_wrap.model_patcher.LanPaint_Tamed
        model_k.beta_scale = model_wrap.model_patcher.LanPaint_BetaScale
        model_k.step_size_schedule = model_wrap.model_patcher.LanPaint_StepSizeSchedule
        model_k.step_time_schedule = model_wrap.model_patcher.LanPaint_StepTimeSchedule
        model_k.start_sigma = model_wrap.model_patcher.LanPaint_StartSigma
        model_k.end_sigma = model_wrap.model_patcher.LanPaint_EndSigma
        model_k.LanPaint_Lambda_Schedule = model_wrap.model_patcher.LanPaint_Lambda_Schedule
        model_k.LanPaint_Cap_Sigma = model_wrap.model_patcher.LanPaint_Cap_Sigma
        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))
        #if not inpainting, after noise_scaling, noise = noise * sigma, which is the noise added to the clean latent image in the variance exploding diffusion model notation.
        #if inpainting, after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        #print("LanPaint KSampler call sampler_function", self.sampler_function)
        # The main loop!
        #print("##########")
        #print("Sampling with ", self.sampler_function)
        #print("##########")
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        #print("LanPaint KSampler end sampler_function")
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples

@contextmanager
def override_sample_function():
    original_outer_sample = comfy.samplers.CFGGuider.outer_sample
    comfy.samplers.CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample

    original_predict_noise = comfy.samplers.CFGGuider.predict_noise
    comfy.samplers.CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise

    original_sample = comfy.samplers.KSAMPLER.sample
    comfy.samplers.KSAMPLER.sample = KSAMPLER.sample

    try:
        yield
    finally:
        comfy.samplers.KSAMPLER.sample = original_sample
        comfy.samplers.CFGGuider.predict_noise = original_predict_noise
        comfy.samplers.CFGGuider.outer_sample = original_outer_sample


class LanPaint_UpSale_LatentNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "scale": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"


    CATEGORY = "latent/inpaint"

    def set_mask(self, samples, scale):
        s = samples.copy()
        samples = s['samples']
        # generate a mask with every scaleth pixel set to 1
        mask = torch.zeros(samples.shape[0], 1, samples.shape[2], samples.shape[3], device=samples.device) + 1
        mask[:, :, ::scale, ::scale] = 0
        s["noise_mask"] = mask
        return (s,)

KSAMPLER_NAMES = ["euler", "dpmpp_2m", "uni_pc"]

class LanPaint_KSampler():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (KSAMPLER_NAMES, {"tooltip": "Recommended: euler."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 20, "tooltip": "The number of steps for the Langevin dynamics, representing the turns of thinking per step."}),  
                "LanPaint_EndSigma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.01, "tooltip": "The noise level at which the thinking process stops. Higher value give less thinking but helps to deal with blurring when turns of thinking is too high."}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler. Recommend steps 50, LanPaint NumSteps 1-20 depending on the difficulty of task. LanPaint_EndSigma = 3.0 for anime style, 0.6 for realistic style. For more information, visit https://github.com/scraed/LanPaint", "multiline": True}),
                  }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, LanPaint_StepSize=0.05, LanPaint_NumSteps=5, LanPaint_EndSigma = 3., LanPaint_Info=""):
        model.LanPaint_StepSize = 0.5
        model.LanPaint_Lambda = 8.0
        model.LanPaint_Beta = 1.2
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = 5.
        model.LanPaint_Alpha = 0.9
        model.LanPaint_Tamed = 1.
        model.LanPaint_BetaScale = "shrink"
        model.LanPaint_StepSizeSchedule = "linear"
        model.LanPaint_StepTimeSchedule = "shrink"
        model.LanPaint_StartSigma = 20.
        model.LanPaint_EndSigma = LanPaint_EndSigma
        model.LanPaint_cfg_BIG = -0.5
        model.LanPaint_Lambda_Schedule = "const"
        with override_sample_function():
            return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
class LanPaint_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (KSAMPLER_NAMES, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "The number of steps for the Langevin dynamics, representing the turns of thinking per step."}),
                "LanPaint_Lambda": ("FLOAT", {"default": 8., "min": 0.1, "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The lambda parameter for the bidirectional guidance. Higher values align with known regions more closely, but may result in instability."}),
                "LanPaint_StepSize": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1., "step": 0.01, "round": 0.001, "tooltip": "The step size for the Langevin dynamics. Higher values result in faster convergence but may be unstable."}),
                "LanPaint_Beta": ("FLOAT", {"default": 1.2, "min": 0.0001, "max": 5, "step": 0.1, "round": 0.1, "tooltip": "The beta parameter for the bidirectional guidance. Scale the step size for the known region independently for the Langevin dynamics. Higher values result in faster convergence but may be unstable."}),
                "LanPaint_Friction": ("FLOAT", {"default": 5., "min": 0., "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The friction parameter for the underdamped Langevin dynamics, higher values result in faster convergence but may be unstable."}),
                "LanPaint_Alpha": ("FLOAT", {"default": 0.9, "min": 0.0001, "max": 1., "step": 0.1, "round": 0.1, "tooltip": "The (rescaled) alpha parameter for the HFHR langevin dynamics, mixes Langevin dynamics and underdamped Langevin dynamics with a friction term. 0 corresponds to Langevin dynamics, 1 corresponds to underdamped Langevin dynamics."}),
                "LanPaint_Tamed": ("FLOAT", {"default": 1., "min": 0.000, "max": 20., "step": 0.1, "round": 0.1, "tooltip": "The tame strength for the noise, normalize and projects the noise onto unit sphere to enhance stability."}),
                "LanPaint_BetaScale": (["shrink", "fixed", "dual_shrink", "back_shrink"], {"default": "shrink", "tooltip": "The beta scale, determines how the beta parameter changes over time. Shrink: beta = beta * (1 - alpha bar) ** 0.5; Fixed: beta = beta; Dual_shrink: beta = beta * (1 - alpha bar) ** 0.5 *  alpha bar ** 0.5; Back_shrink: beta = beta *  alpha bar ** 0.5; Alpha bar: the alpha cumprod."}),
                "LanPaint_StepSizeSchedule": (["const", "linear"], {"default": "linear", "tooltip": "The step size schedule for the Langevin dynamics, const: constant step size, linear: linearly decreasing step size."}),
                "LanPaint_StepTimeSchedule": (["shrink", "dual_shrink", "follow_sampler"], {"default": "shrink", "tooltip": "The step size schedule for the first step of Langevin dynamics during diffusion sampling, shrink: step size = step size * (1 - alpha bar) ** 0.5; Dual_shrink: step size = step size * (1 - alpha bar) ** 0.5 *  alpha bar ** 0.5; Follow_sampler: scale with the sampler step size."}),
                "LanPaint_StartSigma": ("FLOAT", {"default": 20., "min": 0.0001, "max": 20.0, "step": 0.1, "round": 0.1, "tooltip": "Start 'thinking' with Langevin dynamics at this sigma value."}),
                "LanPaint_EndSigma": ("FLOAT", {"default": 3., "min": 0.000, "max": 20.0, "step": 0.1, "round": 0.1, "tooltip": "Stop 'thinking' with Langevin dynamics at this sigma value."}),
                "LanPaint_cfg_BIG": ("FLOAT", {"default": -0.5, "min": -20, "max": 20.0, "step": 0.1, "round": 0.1, "tooltip": "The CFG scale used in the bidirectional guidance (for the known region only). Higher value results in more closely matching the known region."}),
                "LanPaint_Lambda_Schedule": (["const", "shrink"], {"default": "const", "tooltip": "The lambda schedule for the bidirectional guidance, const: constant lambda, shrink: decreasing lambda."}),
                "LanPaint_Cap_Sigma": ("FLOAT", {"default": 1., "min": 0.0001, "max": 20.0, "step": 0.1, "round": 0.1, "tooltip": "Cap the stepsize to scheduler step size below this noise level."}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler Advanced. For difficult tasks, first try increasing steps, LanPaint_NumSteps, and LanPaint_cfg_BIG. Then try increase LanPaint_Lambda or LanPaint_StepSize. Decrease LanPaint_Friction if you want to obtain good results with fewer turns of thinking (LanPaint_NumSteps) at the risk of irregular behavior. Increase LanPaint_Tamed or LanPaint_Alpha can suppress irregular behavior. For more information, visit https://github.com/scraed/LanPaint", "multiline": True}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, LanPaint_StepSize=0.05, LanPaint_Lambda=5, LanPaint_Beta=1, LanPaint_NumSteps=5, LanPaint_Friction=5, LanPaint_Alpha=1, LanPaint_Tamed=0., LanPaint_BetaScale="fixed", LanPaint_StepSizeSchedule = "const", LanPaint_StepTimeSchedule = "shrink",  LanPaint_StartSigma=20, LanPaint_EndSigma=0, LanPaint_cfg_BIG = 5., LanPaint_Lambda_Schedule="const", LanPaint_Cap_Sigma = 0., LanPaint_Info=""):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        model.LanPaint_StepSize = LanPaint_StepSize
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = LanPaint_Beta
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = LanPaint_Friction
        model.LanPaint_Alpha = LanPaint_Alpha
        model.LanPaint_Tamed = LanPaint_Tamed
        model.LanPaint_BetaScale = LanPaint_BetaScale
        model.LanPaint_StepSizeSchedule = LanPaint_StepSizeSchedule
        model.LanPaint_StepTimeSchedule = LanPaint_StepTimeSchedule
        model.LanPaint_StartSigma = LanPaint_StartSigma
        model.LanPaint_EndSigma = LanPaint_EndSigma
        model.LanPaint_cfg_BIG = LanPaint_cfg_BIG
        model.LanPaint_Lambda_Schedule = LanPaint_Lambda_Schedule
        model.LanPaint_Cap_Sigma = LanPaint_Cap_Sigma

        with override_sample_function():
            return nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LanPaint_KSampler": LanPaint_KSampler,
    "LanPaint_KSamplerAdvanced": LanPaint_KSamplerAdvanced,
#    "LanPaint_UpSale_LatentNoiseMask": LanPaint_UpSale_LatentNoiseMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LanPaint_KSampler": "LanPaint KSampler",
    "LanPaint_KSamplerAdvanced": "LanPaint KSampler (Advanced)",
#    "LanPaint_UpSale_LatentNoiseMask": "LanPaint UpSale Latent Noise Mask"
}
