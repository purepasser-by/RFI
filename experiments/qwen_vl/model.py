from __future__ import annotations

from collections import namedtuple
from typing import Callable, Literal, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from ema_pytorch import EMA
from torch import Tensor, pi
from torch.nn import Module
from torchdiffeq import odeint
import torch.nn as nn
import math

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_timescale=10000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_timescale = max_timescale

    def forward(self, t):
        position = t.unsqueeze(1)  # Shape: (batch_size, 1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(self.max_timescale) / self.embedding_dim)).to(t.device)
        
        pos_embedding = torch.zeros(t.shape[0], self.embedding_dim, device=t.device)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding  # Shape: (batch_size, embedding_dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x += residual
        return self.relu(x)


class LinearUNet(nn.Module):
    def __init__(self, hid_dim=4096, depth=4, feature_scale=0.5, time_embedding_dim=128):
        super(LinearUNet, self).__init__()
        self.depth = depth
        self.time_embedding_dim = time_embedding_dim
        self.hid_dim = hid_dim 
        
        # Time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(embedding_dim=time_embedding_dim)

        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = int(hid_dim * (feature_scale ** i)) + time_embedding_dim
            out_dim = int(hid_dim * (feature_scale ** (i + 1)))
            self.down_layers.append(ResidualBlock(in_dim, out_dim))

        # Bottleneck layer
        self.bottleneck = ResidualBlock(int(hid_dim * (feature_scale ** depth)) + time_embedding_dim, int(hid_dim * (feature_scale ** depth)))
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_dim = int(hid_dim * (feature_scale ** (i + 1))) * 2 + time_embedding_dim
            out_dim = int(hid_dim * (feature_scale ** i))
            self.up_layers.append(ResidualBlock(in_dim, out_dim))

        # Final output layer
        self.final_layer = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, times):
        # Compute time embedding
        time_emb = self.time_embedding(times)

        down_features = []

        # Downsampling
        for down in self.down_layers:
            x = torch.cat([x, time_emb], dim=-1)
            x = down(x)
            down_features.append(x)
        
        # Bottleneck
        x = torch.cat([x, time_emb], dim=-1)
        x = self.bottleneck(x)

        # Upsampling
        for up in self.up_layers:
            down_feat = down_features.pop()
            x = torch.cat([x, down_feat, time_emb], dim=-1)
            x = up(x)
        
        # Final layer
        x = self.final_layer(x)
        
        return x

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse'
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: Tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: Tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: Tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        if isinstance(model, dict):
            # model = Unet(**model)
            model = LinearUNet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)

        self.predict = predict

        # automatically default to a working setting for predict epsilon

        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn

        if loss_fn == 'mse':
            loss_fn = MSELoss()
        else:
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules

        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, interp, *, times, eps = 1e-10):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """

        batch = interp.shape[0]

        # prepare maybe time conditioning for model

        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})

        output = model(interp, **model_kwargs)

        # depending on objective, derive flow

        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            x_0 = output
            padded_times = append_dims(times, interp.ndim - 1)

            flow = (interp - x_0) / padded_times.clamp(min = eps)

        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        hidden_states: Tensor | None = None,
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)

            _, flow = self.predict_flow(model, x, times = t, **model_kwargs)

            flow = maybe_clip_flow(flow)

            return flow

        # start with current y_lose
        assert exists(hidden_states), 'you need to pass in hidden states'
        z_0 = hidden_states
        # noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, z_0, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        #! return self.data_unnormalize_fn(sampled_data)  #! don't know whether to unnormalize or not
        return sampled_data

    def forward(
        self,
        y_win,
        y_lose,
        noise: Tensor | None = None,
        return_loss_breakdown = False,
        **model_kwargs
    ):
        assert y_win.shape == y_lose.shape, 'y_win and y_lose must have the same shape'
        batch, *data_shape = y_win.shape

        # data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - y_lose, x1 - y_win

        x_0 = y_lose
        x_1 = y_win
        # noise = default(noise, torch.randn_like(data))

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, x_1.ndim - 1)

        # time needs to be from [0, 1 - delta_time] if using consistency loss

        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time

        def get_noised_and_flows(model, t):

            # maybe noise schedule

            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

            interp = t * x_1 + (1. - t) * x_0

            # the model predicts the flow from the noised data

            flow = x_1 - x_0

            model_output, pred_flow = self.predict_flow(model, interp, times = t)

            # predicted data will be the noised xt + flow * (1. - t)

            pred_data = interp + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_data

        # getting flow and pred flow for main model

        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)

        # if using consistency loss, also need the ema model predicted flow

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_data = get_noised_and_flows(self.ema_model, padded_times + delta_t)

        # determine target, depending on objective

        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # losses

        main_loss = self.loss_fn(output, target, pred_data = pred_data, times = times, data = x_1)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        if self.use_consistency:
            # consistency losses from consistency fm paper - eq (6) in https://arxiv.org/html/2407.02398v1

            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)

            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

        # total loss

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        # loss breakdown

        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)

