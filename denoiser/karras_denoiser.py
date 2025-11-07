import numpy as np
import torch
import torch.nn.functional as F
from piq import LPIPS
from utils.dist_utils import dev

from utils.nn_utils import mean_flat, append_dims, append_zero
from utils.random_utils import get_generator


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDiffusion:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        steps=40,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = steps
        self.interstep_weights = None
        # Track additional rollout decay applied once training leaves the warmup phase.
        # This is controlled externally (e.g. from the trainer) when the discriminator
        # loss plateaus and we want to gradually reduce the rollout horizon.
        self.rollout_decay = 0

    def set_rollout_weighting(self, weights):
        self.interstep_weights = weights

    def set_rollout_decay(self, decay: int):
        """Update the rollout decay applied after the warmup phase.

        Args:
            decay: Number of rollout steps to subtract from the provided
                ``max_rollout`` once the training leaves warmup. Clamped to be
                non-negative.
        """

        if decay is None:
            decay = 0
        self.rollout_decay = max(int(decay), 0)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, self.num_timesteps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        terms = {}

        sigmas_all = self.get_sigmas().to(dev())

        sigmas = sigmas_all[t]

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)

        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def consistency_losses(
            self,
            model,
            x_start,
            num_scales,
            model_kwargs=None,
            target_model=None,
            teacher_model=None,
            teacher_diffusion=None,
            noise=None,
            rollout=10,  # Changed from max_rollout/min_rollout to single rollout parameter
            unnoised_training_epochs=20,
            current_epoch=0,
    ):
        """Compute consistency losses.
        Args:
            model: The student model being trained.
            x_start: Clean input samples.
            num_scales: Number of noise scales in the diffusion schedule.
            model_kwargs: Optional conditioning inputs for the model.
            target_model: Target EMA model for distillation.
            teacher_model: Optional teacher model for Heun updates.
            teacher_diffusion: Diffusion object for teacher model.
            noise: Optional pre-sampled noise.
            rollout: Fixed number of rollout steps to perform for all batch elements
                during the multi-step rollout.
            unnoised_training_epochs: train discriminator for n epochs on
                index==0 and index==num_scales true and false images
                (i.e. full noise and complete denoised) to learn easy
                examples for n epochs
            current_epoch: current epoch of training
        """
        if model_kwargs is None:
            model_kwargs = {}
        if isinstance(model_kwargs, torch.Tensor):
            model_kwargs = {'tensor': model_kwargs}
        if noise is None:
            noise = torch.randn_like(x_start)
        dims = x_start.ndim
        device = x_start.device
        B = x_start.shape[0]
        warmup = (
                unnoised_training_epochs is not None
                and unnoised_training_epochs > 0
                and current_epoch < unnoised_training_epochs
        )

        # ---- helpers ------------------------------------------------------------
        def _index_select_batch(obj, idx_long):
            """Index-select along batch dim for tensors inside nested dict/list/tuple."""
            if obj is None:
                return None
            if torch.is_tensor(obj):
                if obj.dim() == 0:  # scalar
                    return obj
                if obj.size(0) == B:  # batch-aligned tensor
                    return obj.index_select(0, idx_long)
                return obj  # non-batch tensor -> pass through
            if isinstance(obj, dict):
                return {k: _index_select_batch(v, idx_long) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                out = [_index_select_batch(v, idx_long) for v in obj]
                return type(obj)(out)
            return obj

        def _slice_prefix_batch(obj, k):
            """Take first k along batch dim for tensors shaped [B,...]."""
            if obj is None:
                return None
            if torch.is_tensor(obj):
                if obj.dim() == 0:
                    return obj
                if obj.size(0) >= k:
                    return obj[:k]
                return obj
            if isinstance(obj, dict):
                return {k2: _slice_prefix_batch(v2, k) for k2, v2 in obj.items()}
            if isinstance(obj, (list, tuple)):
                out = [_slice_prefix_batch(v, k) for v in obj]
                return type(obj)(out)
            return obj

        # ------------------------------------------------------------------------
        def denoise_fn(x, t, model_kwargs_local):
            return self.denoise(model, x, t, **model_kwargs_local)

        if target_model:
            @torch.no_grad()
            def target_denoise_fn(x, t, model_kwargs_local):
                return self.denoise(target_model, x, t, **model_kwargs_local)
        else:
            raise NotImplementedError("Must have a target model")
        if teacher_model:
            # @torch.no_grad()
            def teacher_denoise_fn(x, t, model_kwargs_local):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs_local)

        # @torch.no_grad()
        def heun_solver(x, t, next_t, x0, model_kwargs_local=None):
            # One step of Heun's method using teacher model
            if teacher_model is None:
                denoised = x0
            else:
                _, denoised = teacher_denoise_fn(x, t, model_kwargs_local)
            d = (x - denoised) / append_dims(t, dims)
            x_next = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoised_next = x0
            else:
                _, denoised_next = teacher_denoise_fn(x_next, next_t, model_kwargs_local)
            d_2 = (x_next - denoised_next) / append_dims(next_t, dims)
            x_next = x + (d + d_2) * append_dims((next_t - t) / 2, dims)
            return x_next

        def heun_solver_consistency(x, t, next_t, x0, model_kwargs_local=None):
            # One step of Heun's method using denoising (consistency) model
            _, denoised = denoise_fn(x, t, model_kwargs_local)
            d = (x - denoised) / append_dims(t, dims)
            x_next = x + d * append_dims(next_t - t, dims)
            _, denoised_next = denoise_fn(x_next, next_t, model_kwargs_local)
            d_2 = (x_next - denoised_next) / append_dims(next_t, dims)
            x_next = x + (d + d_2) * append_dims((next_t - t) / 2, dims)
            return x_next

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0, model_kwargs_local=None):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                _, denoiser = teacher_denoise_fn(x, t, model_kwargs_local)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples

        # --- sample starting indices per item (warm-up forces index 0 = full noise) ---
        if num_scales == 1:
            indices = torch.zeros(B, device=device, dtype=torch.long)
        else:
            if warmup:
                # Start every sample at the noisiest level (index 0 -> t = sigma_max)
                indices = torch.zeros(B, device=device, dtype=torch.long)
            else:
                indices = torch.randint(0, num_scales - 1, (B,), device=device, dtype=torch.long)
        # Precompute sigma schedule once and index it
        idxs_all = torch.arange(num_scales, device=device)
        sigma_sched = (
                (self.sigma_max ** (1 / self.rho) + idxs_all / (num_scales - 1) *
                 (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        )
        # Convert indices to times t and t+1 via the schedule
        t = sigma_sched[indices]
        next_idx = torch.clamp(indices + 1, max=num_scales - 1)
        t2 = sigma_sched[next_idx]
        # Noisy current state
        x_t = x_start + noise * append_dims(t, dims)
        # Store RNG state for reproducibility across the two forward passes below
        dropout_state = torch.get_rng_state()
        model_output, distiller = denoise_fn(x_t, t, model_kwargs)
        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start, model_kwargs).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start, model_kwargs).detach()
        torch.set_rng_state(dropout_state)
        model_output_target, distiller_target = target_denoise_fn(x_t2, t2, model_kwargs)
        distiller_target = distiller_target.detach()
        # Weighted reconstruction loss
        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(distiller_target, size=32, mode="bilinear")
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(distiller_target, size=224, mode="bilinear")
            loss = self.lpips_loss((distiller + 1) / 2.0, (distiller_target + 1) / 2.0) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        terms = {"loss": loss}

        # ============================================================================
        # ✅ OPTION 1: SEPARATE SAMPLING FOR DISCRIMINATOR ROLLOUT
        # ============================================================================
        # Sample indices that guarantee full rollout distance for discriminator
        if warmup:
            disc_start_indices = torch.zeros(B, device=device, dtype=torch.long)
        else:
            # ✅ Calculate the maximum valid starting index that allows full rollout
            # For rollout=9, max_valid_start=1 (indices 0 or 1 allow reaching index 9 or 10)
            max_valid_start = max(0, num_scales - 1 - rollout)

            if max_valid_start == 0:
                # Only index 0 is valid
                disc_start_indices = torch.zeros(B, device=device, dtype=torch.long)
            else:
                # Sample uniformly from valid range [0, max_valid_start]
                disc_start_indices = torch.randint(0, max_valid_start + 1, (B,), device=device, dtype=torch.long)

        # ============================================================================
        # DISCRIMINATOR ROLLOUT SECTION
        # ============================================================================
        if warmup:
            # For warm-up, use full remaining steps from index 0
            rem = torch.clamp((num_scales - 1) - disc_start_indices, min=0)
            rollout_steps = rem

            x_fake = x_start + noise * append_dims(sigma_sched[disc_start_indices], dims)
            t_cur_idx = disc_start_indices.clone()
            t_start_idx = disc_start_indices.clone()
            x0_local = x_start
            kwargs_local = model_kwargs

            max_steps = int(rollout_steps.max().item()) if rollout_steps.numel() > 0 else 0
            for step in range(max_steps):
                k = int((rollout_steps > step).sum().item())
                if k <= 0:
                    break
                if k < 2:
                    break

                cur_idx = t_cur_idx[:k]
                next_idx = torch.minimum(cur_idx + 1, torch.full_like(cur_idx, num_scales - 1))
                t_now = sigma_sched[cur_idx]
                t_next = sigma_sched[next_idx]

                kwargs_step = _slice_prefix_batch(kwargs_local, k)
                x_fake[:k] = heun_solver_consistency(
                    x_fake[:k], t_now, t_next, x0_local[:k], kwargs_step
                )

                t_cur_idx[:k] = next_idx

            t_cur = sigma_sched[t_cur_idx]
            t_start = sigma_sched[t_start_idx]
            x_tn_consistency = x_fake

        else:
            # ✅ Use fixed rollout - no clamping needed since we sampled valid indices!
            rollout_steps = torch.full((B,), rollout, device=device, dtype=torch.long)

            # Sort by rollout steps for efficient processing (all same, but keep structure)
            roll_sorted, order = torch.sort(rollout_steps, descending=True)
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=device)

            # Reorder batch-wise things
            x_t_disc = x_start.index_select(0, order) + noise.index_select(0, order) * append_dims(
                sigma_sched[disc_start_indices.index_select(0, order)], dims
            )
            x_fake = x_t_disc.clone()
            t_cur_idx = disc_start_indices.index_select(0, order).clone()
            t_start_idx = disc_start_indices.index_select(0, order).clone()
            x0_sorted = x_start.index_select(0, order)
            kwargs_sorted = _index_select_batch(model_kwargs, order)

            max_steps = int(roll_sorted[0].item()) if roll_sorted.numel() > 0 else 0
            for step in range(max_steps):
                k = int((roll_sorted > step).sum().item())
                if k <= 0:
                    break
                if k < 2:
                    break

                cur_idx = t_cur_idx[:k]
                next_idx = torch.minimum(cur_idx + 1, torch.full_like(cur_idx, num_scales - 1))
                t_now = sigma_sched[cur_idx]
                t_next = sigma_sched[next_idx]

                kwargs_step = _slice_prefix_batch(kwargs_sorted, k)
                x_fake[:k] = heun_solver_consistency(
                    x_fake[:k], t_now, t_next, x0_sorted[:k], kwargs_step
                )

                t_cur_idx[:k] = next_idx

            # Map back to original batch order
            t_cur_idx = t_cur_idx.index_select(0, inv_order)
            t_start_idx = t_start_idx.index_select(0, inv_order)

            t_cur = sigma_sched[t_cur_idx]
            t_start = sigma_sched[t_start_idx]

            x_fake = x_fake.index_select(0, inv_order)
            x_tn_consistency = x_fake

        # ✅ Create starting noisy image for discriminator
        x_t_rollout_start = x_start + noise * append_dims(t_start, dims)

        # True noisy reference at the final times
        x_tn_true = x_start + noise * append_dims(t_cur, dims)

        # ✅ Calculate actual rollout distances achieved
        actual_rollout_distances = (t_cur_idx - t_start_idx).float()

        terms.update({
            "x_tn_consistency": x_tn_consistency,  # predicted noisy image of CM after rollout
            "x_tn_model_output": model_output,  # predicted denoised image of CM
            "x_tn_target": distiller_target,  # predicted noisy image of Target Model after rollout
            "x_tn_target_model_output": model_output_target,  # predicted denoised image of target model
            "x_tn_true": x_tn_true,  # true noisy image at end of rollout
            "x_t": x_t_rollout_start,  # ✅ noisy image at START of rollout
            "t": t_start,  # ✅ Starting timestep of rollout
            "t_cur": t_cur,  # Ending timestep of rollout
            "x_start": x_start,
            "model_kwargs": model_kwargs,
            "actual_rollout": actual_rollout_distances,  # ✅ Track actual rollout per sample
            # Timestep-based weighting (inverse ramp)
            "t_weight": (((t_cur ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho))) /
                         ((self.sigma_min ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho)))),
        })

        return terms

    def progdist_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        @torch.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(
                next_t - t, dims
            )
            return denoiser

        indices = torch.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t3 = t3**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        denoised_x = denoise_fn(x_t, t)

        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()

        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = torch.abs(denoised_x - target_x)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (denoised_x - target_x) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                denoised_x = F.interpolate(denoised_x, size=224, mode="bilinear")
                target_x = F.interpolate(target_x, size=224, mode="bilinear")
            loss = (
                self.lpips_loss(
                    (denoised_x + 1) / 2.0,
                    (target_x + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, model_kwargs)
        denoised = c_out * model_output + c_skip * x_t

        return model_output, denoised


def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80.0,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, model_kwargs=model_kwargs
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    if isinstance(model_kwargs, torch.Tensor):
        model_kwargs = {'tensor': model_kwargs}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs torch noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates torche noise level (sigma_down) to step down to and torche amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling witorch Euler metorchod steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler metorchod
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling witorch midpoint metorchod steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    model_kwargs=None
):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # if currentepoch > 50 and i == indices[-2]:
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        #
        #     # Plot the image at index 0 on the left
        #     axes[0,0].imshow(x[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[0,0].set_title(model_kwargs['tensor'][0][4:6])
        #     axes[0,0].axis("off")
        #
        #     # Plot the image at index 1 on the right
        #     axes[0,1].imshow(x[1, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[0,1].set_title(model_kwargs['tensor'][1][4:6])
        #     axes[0,1].axis("off")
        #
        #     # Plot the image at index 0 on the left
        #     axes[1,0].imshow(x[2, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[1,0].set_title(model_kwargs['tensor'][2][4:6])
        #     axes[1,0].axis("off")
        #
        #     # Plot the image at index 1 on the right
        #     axes[1,1].imshow(x[3, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[1,1].set_title(model_kwargs['tensor'][3][4:6])
        #     axes[1,1].axis("off")
        #
        #     plt.tight_layout()
        #     plt.show()

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler metorchod
            x = x + d * dt
        else:
            # Heun's metorchod
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

    return x


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algoritorchm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint metorchod, where torche midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip torche zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@torch.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_ortorchogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = torch.from_numpy(obtain_ortorchogonal_matrix()).to(dev()).to(torch.float32)
    mask = torch.zeros(*x.shape[1:], device=dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = torch.einsum("bchw,cd->bdhw", x0, Q)
        x1 = torch.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = torch.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, torch.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@torch.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image witorch a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for torche image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw torche letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert torche image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_torch = torch.from_numpy(img_np).to(dev())

    mask = torch.zeros(*x.shape, device=dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_torch > 0.5] = 1.0
    mask[1::2, :, img_torch < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -torch.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@torch.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_ortorchogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = torch.from_numpy(obtain_ortorchogonal_matrix()).to(dev()).to(torch.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = torch.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = torch.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = torch.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images
