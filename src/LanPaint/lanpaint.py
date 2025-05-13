import torch
from .utils import *
from functools import partial
class LanPaint():
    def __init__(self, Model, NSteps, Friction, Lambda, Alpha, Beta, StepSize, EndSigma=0, StepSizeSchedule = "follow_sampler", BetaSchedule = "const", IteStepSchedule="const", CapSigma = 1., IS_FLUX = False, IS_FLOW = False):
        self.n_steps = NSteps
        self.chara_lamb = Lambda
        self.IS_FLUX = IS_FLUX
        self.IS_FLOW = IS_FLOW
        self.step_size = StepSize
        self.inner_model = Model
        self.alpha = Alpha
        self.beta_scale = BetaSchedule
        self.step_size_schedule = IteStepSchedule
        self.step_time_schedule = StepSizeSchedule
        self.end_sigma = EndSigma
        self.friction = Friction
        self.LanPaint_Cap_Sigma = CapSigma
        self.chara_beta = Beta
        
    def __call__(self, x, latent_image, noise, sigma, Sigmas, latent_mask, current_times, model_options, seed):
        self.VE_Sigmas = Sigmas
        self.latent_image = latent_image
        self.noise = noise
        return self.LanPaint(x, sigma, latent_mask, current_times, model_options, seed, self.IS_FLUX, self.IS_FLOW)
    def LanPaint(self, x, sigma, latent_mask, current_times, model_options, seed, IS_FLUX, IS_FLOW):
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
            step_size = self.step_size * (1 -  abt ) ** 0.5



        step_size = step_size[:, None, None, None]
        # self.inner_model.inner_model.scale_latent_inpaint returns variance exploding x_t values
        # This is the replace step
        x = x * (1 - latent_mask) +  self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image)* latent_mask
        
        if IS_FLUX or IS_FLOW:
            x_t = x * ( 1 + VE_Sigma[:, None,None,None])
        else:
            x_t = x #/ ( 1+sigma**2 )**0.5 # switch to variance perserving x_t values

        ############ LanPaint Iterations Start ###############
        # after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        args = None
        for i in range(self.n_steps):
            if torch.mean(VE_Sigma) < self.end_sigma:
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
        out = out * (1-latent_mask) + self.latent_image * latent_mask
        return out
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

        if self.IS_FLUX or self.IS_FLOW:
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
