from inspect import cleandoc
import inspect
# import nodes.py
import comfy
import nodes
import latent_preview
from functools import partial
from comfy.utils import repeat_to_batch_size
# Monkey patch comfy.samplers module by importing with absolute package path
exec(inspect.getsource(comfy.samplers).replace("from .", "from comfy."))

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
CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample

class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        # x is x_t in the notation of variance exploding diffusion model, x_t = x_0 + sigma * noise
        # sigma is the noise level
        # print what is inside model_options
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            denoise_mask = (denoise_mask > 0.5).float()

            latent_mask = 1 - denoise_mask
            #latent_mask = (latent_mask > 0.5).long()
            

            abt = 1/( 1+sigma**2 )
            step_size = self.step_size * (1 - abt) ** 0.5

            
            current_times = (sigma, abt)

            # self.inner_model.inner_model.scale_latent_inpaint returns variance exploding x_t values
            x = x * (1 - latent_mask) +  self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image)* latent_mask
            x_t = x / ( 1+sigma**2 )**0.5 # switch to variance perserving x_t values
            # after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
            args = None
            for i in range(self.n_steps):
                score_func = partial( self.score_model, y = self.latent_image, mask = latent_mask, abt = abt, sigma = sigma, model_options = model_options, seed = seed )
                if self.step_size_schedule == "linear":
                    step_size_i = step_size * (1 - i/(self.n_steps) )
                else:
                    step_size_i = step_size 
                x_t, args = self.langevin_dynamics(x_t, score_func , latent_mask, step_size_i , current_times, sigma_x = self.sigma_x(abt), sigma_y = self.sigma_y(abt), args = args)  
            x = x_t * ( 1+sigma**2 )**0.5
            # out is x_0
            out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
            #out = out * denoise_mask + self.latent_image * latent_mask
        else:
            out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        return out
    def mid_times(self, current_times, step_size):
        sigma, abt = current_times
        tt = torch.log(1+sigma**2)
        tt_mid = torch.max( tt - step_size, tt*0 )
        sigma_mid = (torch.exp(tt_mid) - 1) ** 0.5
        abt_mid = 1/(1+sigma_mid**2)
        return sigma_mid, abt_mid
    def score_model(self, x_t, y, mask, abt, sigma, model_options, seed):
        # the score function for the Langevin dynamics
        lamb = self.chara_lamb
        beta = self.chara_beta * (1-abt)**0.5

        x_0 = self.inner_model(x_t*( 1+sigma**2 )**0.5, sigma, model_options=model_options, seed=seed)
        e_t = (x_t - abt ** 0.5 * x_0) / (1 - abt) ** 0.5
        score_x = -e_t
        score_y = (- (1 + lamb) * ( x_t - abt**0.5 * y ) /(1 - abt)**0.5 + lamb *  e_t)
        return score_x * (1 - mask) + score_y * mask
    def sigma_x(self, abt):
        # the time scale for the x_t update
        return 1
    def sigma_y(self, abt):
        # the time scale for the y_t update
        if self.beta_scale == "shrink":
            beta = self.chara_beta * (1-abt)**0.5
        elif self.beta_scale == "dual_shrink":
            beta = self.chara_beta * (1-abt)**0.5 * abt ** 0.5
        elif self.beta_scale == "back_shrink":
            beta = self.chara_beta * abt ** 0.5
        else:
            beta = self.chara_beta
        return beta
    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):
        # -------------------------------------------------------------------------
        # Unpack current times parameters (sigma and abt)
        sigma, abt = current_times

        # Compute time step (dtx, dty) for x and y branches.
        dtx = 2 * step_size * sigma_x
        dty = 2 * step_size * sigma_y

        # -------------------------------------------------------------------------
        # Define friction parameter Gamma_hat for each branch.
        # Using dtx**0 provides a tensor of the proper device/dtype.
        Gamma_hat_x = self.friction * dtx / (1e-4+ 2 * sigma_x * 0.1 * (1-abt)**0.5)
        Gamma_hat_y = self.friction * dty / (1e-4+ 2 * sigma_y * 0.1 * (1-abt)**0.5)

        # Get mid time parameters (sigma_mid and abt_mid) for each branch.
        sigma_mid_x, abt_mid_x = self.mid_times(current_times, dtx)
        sigma_mid_y, abt_mid_y = self.mid_times(current_times, dty)

        # -------------------------------------------------------------------------
        # A: Update epsilon (score estimate and noise initialization)
        # -------------------------------------------------------------------------
        # Compute the score-based epsilon (scaled as sqrt(1-abt))
        score_model = score(x_t)
        eps_model = -score_model 



        # Initialize epsilon and Z if not provided in args.
        if args is None:
            eps = eps_model
            Z = torch.randn_like(x_t)
        else:
            eps, Z = args

        # -------------------------------------------------------------------------
        # B: Update epsilon mean dynamics and compute the mid-point in z-space.
        # -------------------------------------------------------------------------
        # Compute the weighted combination term for epsilon mean update:
        # term = (2/Γ_hat)*(1-exp(-0.5*Γ_hat))
        term_x = 2.0 / (Gamma_hat_x + 1e-4) * (1 - torch.exp(-0.5 * Gamma_hat_x))
        term_y = 2.0 / (Gamma_hat_y + 1e-4) * (1 - torch.exp(-0.5 * Gamma_hat_y))
        eps_bar_x = term_x * eps + (1 - term_x) * eps_model
        eps_bar_y = term_y * eps + (1 - term_y) * eps_model
        # Combine branches according to mask.
        eps_bar = eps_bar_x * (1 - mask) + eps_bar_y * mask

        # Form the denoised epsilon using self.alpha (assumed to be 1/Ψ)
        eps_denoise = self.alpha * eps_bar + (1 - self.alpha) * eps_model


        # tamed
        eps_model_x = eps_denoise* (1 - mask)
        eps_model_x = eps_model_x* (torch.sum(1 - mask, dim = (1,2,3))/torch.sum(eps_model_x**2, dim = (1,2,3))) **0.5 ** torch.minimum(self.tamed*(dtx),sigma**0)#/( 1 + self.tamed*(sigma - sigma_mid_x) * (torch.sum(eps_model_x**2)/torch.sum((1 - mask)))**0.5 )
        eps_model_y = eps_denoise* mask
        eps_model_y = eps_model_y* (torch.sum(mask, dim = (1,2,3))/torch.sum(eps_model_y**2, dim = (1,2,3))) **0.5 ** torch.minimum(self.tamed*(dty),sigma**0)#/( 1 + self.tamed*(sigma - sigma_mid_y) * (torch.sum(eps_model_y**2)/torch.sum(mask))**0.5 )
        eps_denoise = eps_model_x * (1 - mask) + eps_model_y * mask


        # Update the mean epsilon for the next step:
        eps_x = eps * torch.exp(-0.5 * Gamma_hat_x) + eps_model * (1 - torch.exp(-0.5 * Gamma_hat_x))
        eps_y = eps * torch.exp(-0.5 * Gamma_hat_y) + eps_model * (1 - torch.exp(-0.5 * Gamma_hat_y))
        eps = eps_x * (1 - mask) + eps_y * mask

        # Transform x to z using z = x * sqrt(1+sigma^2)
        z_t = x_t * (1 + sigma**2) ** 0.5

        # Compute the mid-point update in z-space for each branch:
        z_mid_x = z_t + eps_denoise * (sigma_mid_x - sigma)
        z_mid_y = z_t + eps_denoise * (sigma_mid_y - sigma)
        z_mid = z_mid_x * (1 - mask) + z_mid_y * mask

        # -------------------------------------------------------------------------
        # C: Update noise terms and finalize the x update.
        # -------------------------------------------------------------------------
        # Generate auxiliary noise terms.
        Z_q     = torch.randn_like(x_t)
        Z_q_avg = torch.randn_like(x_t)
        Z_z     = torch.randn_like(x_t)

        # Update Z for each branch:
        Z_x = torch.exp(-0.5 * Gamma_hat_x) * Z + (1 - torch.exp(-Gamma_hat_x)) ** 0.5 * Z_q
        Z_y = torch.exp(-0.5 * Gamma_hat_y) * Z + (1 - torch.exp(-Gamma_hat_y)) ** 0.5 * Z_q
        Z_next = Z_x * (1 - mask) + Z_y * mask

        # Compute the combined noise update following the scheme:
        Z_comb_x = (
            (1 - torch.exp(-Gamma_hat_x / 2)) / torch.sqrt(Gamma_hat_x + 1e-4) *
            (Z + torch.sqrt(torch.tanh(Gamma_hat_x / 4)) * Z_q)
            + torch.sqrt(1 - (4 / (Gamma_hat_x + 1e-4)) * torch.tanh(Gamma_hat_x / 4)) * Z_q_avg
        )
        Z_comb_y = (
            (1 - torch.exp(-Gamma_hat_y / 2)) / torch.sqrt(Gamma_hat_y + 1e-4) *
            (Z + torch.sqrt(torch.tanh(Gamma_hat_y / 4)) * Z_q)
            + torch.sqrt(1 - (4 / (Gamma_hat_y + 1e-4)) * torch.tanh(Gamma_hat_y / 4)) * Z_q_avg
        )
        Z_comb = Z_comb_x * (1 - mask) + Z_comb_y * mask

        # Combine with an additional noise term using self.alpha.
        Z_comb = self.alpha ** 0.5 * Z_comb + (1 - self.alpha) ** 0.5 * Z_z

        # Compute the change in sigma (dsigma = sqrt(sigma^2 - sigma_mid^2)).
        dsigma_x = sigma * torch.sqrt(torch.max(1 - (sigma_mid_x / sigma) ** 2, sigma * 0))
        dsigma_y = sigma * torch.sqrt(torch.max(1 - (sigma_mid_y / sigma) ** 2, sigma * 0))
        dsigma = dsigma_x * (1 - mask) + dsigma_y * mask

        # Final z update.
        z_final = z_mid + Z_comb * dsigma

        # Transform back to x-space: x = z / sqrt(1+sigma^2)
        x_t = z_final / (1 + sigma**2) ** 0.5

        return x_t, (eps, Z_next)
# Custom sampler class extending ComfyUI's KSAMPLER for LanPaint
class KSAMPLER(comfy.samplers.KSAMPLER):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        #noise here is a randn noise from comfy.sample.prepare_noise
        #latent_image is the latent image as input of the KSampler node. For inpainting, it is the masked latent image. Otherwise it is zero tensor.
        extra_args["denoise_mask"] = denoise_mask
        #print("model.LanPaint_StepSize", model_wrap.inner_model.LanPaint_StepSize)
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise
        model_k.step_size = model_wrap.model_patcher.LanPaint_StepSize
        model_k.chara_lamb = model_wrap.model_patcher.LanPaint_Lambda
        model_k.chara_beta = model_wrap.model_patcher.LanPaint_Beta
        model_k.n_steps = model_wrap.model_patcher.LanPaint_NumSteps
        model_k.friction = model_wrap.model_patcher.LanPaint_Friction
        model_k.alpha = model_wrap.model_patcher.LanPaint_Alpha
        model_k.tamed = model_wrap.model_patcher.LanPaint_Tamed
        model_k.beta_scale = model_wrap.model_patcher.LanPaint_BetaScale
        model_k.step_size_schedule = model_wrap.model_patcher.LanPaint_StepSizeSchedule
        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))
        #if not inpainting, after noise_scaling, noise = noise * sigma, which is the noise added to the clean latent image in the variance exploding diffusion model notation.
        #if inpainting, after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        print("LanPaint KSampler call sampler_function", self.sampler_function)
        # The main loop!
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        print("LanPaint KSampler end sampler_function")
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples

# Monkey patch ComfyUI's sample function to use our custom KSAMPLER
comfy_sample_sample = inspect.getsource(comfy.sample.sample).replace("def sample(", "def comfy_sample_sample(").replace("comfy.samplers.KSampler", "KSampler")
exec(comfy_sample_sample)
# Modify nodes.common_ksampler function to use our patched sample function
common_ksampler = inspect.getsource(nodes.common_ksampler).replace("comfy.sample.sample", "comfy_sample_sample")
exec(common_ksampler)


KSAMPLER_NAMES = ["euler", "dpmpp_2m", "uni_pc"]

class LanPaint_KSampler():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (KSAMPLER_NAMES, {"tooltip": "Recommended: euler."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 20, "tooltip": "The number of steps for the Langevin dynamics."}),
                "LanPaint_Lambda": ("FLOAT", {"default": 4, "min": 0.1, "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The lambda parameter for the bidirectional guidance."}),  
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler. For more information, visit https://github.com/scraed/LanPaint", "multiline": True}),
                  }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, LanPaint_StepSize=0.05, LanPaint_Lambda=5, LanPaint_NumSteps=5, LanPaint_Info=""):
        model.LanPaint_StepSize = 0.1
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = 2
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = 20
        model.LanPaint_Alpha = 1.
        model.LanPaint_Tamed = 6
        model.LanPaint_BetaScale = "fixed"
        model.LanPaint_StepSizeSchedule = "const"
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
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
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 20, "tooltip": "The number of steps for the Langevin dynamics."}),
                "LanPaint_Lambda": ("FLOAT", {"default": 4, "min": 0.1, "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The lambda parameter for the bidirectional guidance."}),
                "LanPaint_StepSize": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 0.5, "step": 0.01, "round": 0.001, "tooltip": "The step size for the Langevin dynamics."}),
                "LanPaint_Beta": ("FLOAT", {"default": 2, "min": 0.0001, "max": 5, "step": 0.1, "round": 0.1, "tooltip": "The beta parameter for the bidirectional guidance."}),
                "LanPaint_Friction": ("FLOAT", {"default": 20, "min": 1., "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The friction parameter for the Langevin dynamics."}),
                "LanPaint_Alpha": ("FLOAT", {"default": 1, "min": 0.0001, "max": 1., "step": 0.1, "round": 0.1, "tooltip": "The alpha parameter for the Langevin dynamics."}),
                "LanPaint_Tamed": ("FLOAT", {"default": 6, "min": 0.000, "max": 10., "step": 0.1, "round": 0.1, "tooltip": "The tame strength"}),
                "LanPaint_BetaScale": (["shrink", "fixed", "dual_shrink", "back_shrink"], {"default": "fixed", "tooltip": "The beta scale"}),
                "LanPaint_StepSizeSchedule": (["const", "linear"], {"default": "const", "tooltip": "The step size schedule"}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler Advanced. For more information, visit https://github.com/scraed/LanPaint", "multiline": True}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, LanPaint_StepSize=0.05, LanPaint_Lambda=5, LanPaint_Beta=1, LanPaint_NumSteps=5, LanPaint_Friction=5, LanPaint_Alpha=1, LanPaint_Tamed=0., LanPaint_BetaScale="fixed", LanPaint_StepSizeSchedule = "const",LanPaint_Info=""):
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
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)
class LanPaint_VAEEncodeForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("VAE", ), "mask": ("MASK", ), 
        "grow_mask_by": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
        #"Keep_Luminance": (["disable", "enable"], ),
        #"Keep_Red": (["disable", "enable"], ),
        #"Keep_Lime": (["disable", "enable"], ),
        #"Keep_Structure": (["disable", "enable"], ),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/inpaint"

    def encode(self, vae, pixels, mask, grow_mask_by=6,): #Keep_Luminance="disable", Keep_Red="disable", Keep_Lime="disable", Keep_Structure="disable"):
        x = (pixels.shape[1] // vae.downscale_ratio) * vae.downscale_ratio
        y = (pixels.shape[2] // vae.downscale_ratio) * vae.downscale_ratio
        mask = (mask > 0.5 ).float()
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="nearest-exact")
        
        # print the range of mask values
        print("mask range", mask.min(), mask.max())
        
        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % vae.downscale_ratio) // 2
            y_offset = (pixels.shape[2] % vae.downscale_ratio) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding="same"), 0, 1)
        mask_erosion = (mask_erosion > 0.).float()
        #m = (1.0 - mask.round()).squeeze(1)
        #for i in range(3):
        #    pixels[:,:,:,i] -= 0.5
        #    pixels[:,:,:,i] *= m
        #    pixels[:,:,:,i] += 0.5
        t = vae.encode(pixels)
        dims = len(t.shape) - 2
        mask_erosion = mask_erosion.repeat((1, t.shape[1]) + (1,) * dims)[:,:t.shape[1]]
        # if Keep_Luminance == "enable":
        #     mask_erosion[:,0,:,:] *= 0

        # if Keep_Red == "enable":
        #     mask_erosion[:,1,:,:] *= 0

        # if Keep_Lime == "enable":
        #     mask_erosion[:,2,:,:] *= 0

        # if Keep_Structure == "enable":
        #     mask_erosion[:,3:,:,:] *= 0
        return ({"samples":t, "noise_mask": (mask_erosion.round())}, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LanPaint_KSampler": LanPaint_KSampler,
    "LanPaint_KSamplerAdvanced": LanPaint_KSamplerAdvanced,
    "LanPaint_VAEEncodeForInpaint": LanPaint_VAEEncodeForInpaint,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LanPaint_KSampler": "LanPaint_KSampler",
    "LanPaint_KSamplerAdvanced": "LanPaint_KSamplerAdvanced"
}
