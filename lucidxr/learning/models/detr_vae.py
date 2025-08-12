# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""

from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder

from lucidxr.learning.act_config import ACT_Config
from lucidxr.learning.models.RunningNormLayer import RunningNormLayer, make_norm_denorm_layers
from lucidxr.learning.models.transformer import Transformer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        *,
        backbones: Dict[str, nn.Module],
        transformer,
        encoder,
        obs_dim,
        action_dim,
        latent_dim,
        num_queries,
        image_keys,
        normalize_actions=False,
        normalize_obs=False,
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            obs_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            not used - aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

        """
        super().__init__()
        self.num_queries = num_queries
        self.image_keys = sorted(image_keys)
        self.transformer: Transformer = transformer
        self.encoder: TransformerEncoder = encoder
        self.normalize_actions = normalize_actions
        self.normalize_obs = normalize_obs

        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        # what is the difference between this and the pos_table (position encoding)?
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.backbones: dict = None
        if backbones is not None and len(backbones) > 0:
            first, *_ = backbones.values()
            self.input_proj = nn.Conv2d(first.num_channels, hidden_dim, kernel_size=1)
            # it is a dictionary! - Ge
            self.backbones = backbones
            self.backbones_list = nn.ModuleList(backbones.values())
        self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)
        # else:
        #     # input_dim = 14 + 7 # robot_state + env_state
        #     raise NotImplementedError
        #     self.input_proj_robot_state = nn.Linear(qpos_dim, hidden_dim)
        #     self.input_proj_env_state = nn.Linear(env_state_dim, hidden_dim)
        #     self.pos = torch.nn.Embedding(2, hidden_dim)
        #     self.backbones = None

        # encoder extra parameters
        self.latent_dim: int = latent_dim  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(obs_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))  # should be > [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent

        # these are not learned via gradient descent.
        if normalize_obs:
            self.qpos_norm = RunningNormLayer([obs_dim])
        if normalize_actions:
            # the magic switch is used to make loading work with the action_norm, will be removed in the future. - Ge
            self.action_norm, self.action_denorm = make_norm_denorm_layers([action_dim])

    def forward(self, *, observation, image, actions=None, is_pad=None):
        """
        qpos: batch, seq, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        a_hat_norm, a_norm = None, None
        is_training = actions is not None  # train or val
        bs, *_ = observation.shape

        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            if self.normalize_actions:
                actions = self.action_norm(actions)  # (bs, seq, action_dim)
                a_norm = actions
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)

            if self.normalize_obs:
                pass
                observation = self.qpos_norm(observation)  # (bs, seq, qpos_dim)

            qpos_embed = self.encoder_joint_proj(observation)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, dim=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(observation.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            # mu = torch.zeros_like(mu)  # zero out mu
            logvar = latent_info[:, self.latent_dim:]
            # logvar = torch.zeros_like(logvar)  # zero out logvar
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            # print("Using latent sample from action sequence")
        else:
            mu = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(observation.device)
            logvar = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(observation.device)
            latent_sample = reparametrize(mu, logvar)
            # latent_sample = torch.zeros_like(mu)  # zero out latent sample
            latent_input = self.latent_out_proj(latent_sample)


        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_key in self.image_keys:
                img = image[cam_key]
                bb =  self.backbones[cam_key]
                features, pos = bb(img)
                features = features[-1] # take the last layer feature
                pos = pos[-1]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(observation)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, dim=3)
            pos = torch.cat(all_cam_pos, dim=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[ACT_Config.detr_intermediate_layer]
        else:
            proprio_input = self.input_proj_robot_state(observation)
            hs = self.transformer(None, None, self.query_embed.weight, None, latent_input, proprio_input, self.additional_pos_embed.weight)[ACT_Config.detr_intermediate_layer]

        a_hat = self.action_head(hs)

        if self.normalize_actions:
            a_hat_norm = a_hat
            a_hat = self.action_denorm(a_hat)


        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], [a_hat_norm, a_norm]



class SimpleMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.obs_head = nn.Linear(obs_dim * 3, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim * 1)
        self.qpos_norm = RunningNormLayer([obs_dim])
        self.action_norm, self.action_denorm = make_norm_denorm_layers([action_dim])

    def forward(self, obs, action):
        obs_norm = self.qpos_norm(obs)
        obs_norm = obs_norm.flatten(start_dim=1)
        a_norm = self.action_norm(action)
        h = nn.functional.relu(self.obs_head(obs_norm))
        h = nn.functional.relu(self.hidden_layer(h))
        # h = nn.functional.relu(self.hidden_layer_2(h))
        a_hat_norm = self.action_head(h)
        a_hat_norm = a_hat_norm.reshape(-1, 1, 3)
        a_hat = self.action_denorm(a_hat_norm)
        # a_hat = self.action_head(self.obs_head(obs))
        return a_hat, [a_hat_norm, a_norm]

