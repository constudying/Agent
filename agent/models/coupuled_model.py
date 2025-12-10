#!/usr/bin/env python

# Adapted from InterACT policy for Human-Robot Collaboration
# Original InterACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
# This version: Human behavior modeled as graph structure (GCN), Robot as single 7-DOF arm

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Callable
from torch import Tensor

from .gcn import STSGCN

from dataclasses import dataclass, field

class InterACTConfig:
    """
    Configuration class for Human-Robot Coupled InterACT model.
    """
    chunk_size = 100  # Action prediction chunk size
    n_action_steps = 16  # Number of action steps to predict

    input_shapes = {
        "image": [3, 480, 640],  # Camera observation shape
        "human_graph": [17, 3],  # Human joint coordinates
        "robot_state": [7],      # Robot proprioceptive state
    }
    output_shapes = {
        "human_action": [34],  # Predict 2D joint velocities for human (17 joints x 2D)
        "robot_action": [7],   # Robot action dimension
    }
    # transformer parameters
    num_cls_tokens_human = 3
    num_cls_tokens_robot = 3
    num_cls_tokens_image = 3
    pre_norm = False
    dim_model = 512
    n_heads = 8
    dim_feedforward = 3200
    feedforward_activation = "relu"
    num_blocks = 3

    n_decoder_layers = 4
    n_pre_decoder_layers = 2
    n_post_decoder_layers = 2
    n_sync_decoder_layers = 1

    dropout = 0.1
    kl_weight = 10.0

    # Human GCN parameters
    human_input_channels = 3
    human_input_time_frame = 16
    human_output_time_frame = 16
    human_joints = 17
    human_action_dim = 34  # Predict 2D joint velocities for human
    # Robot parameters
    robot_state_dim = 7
    robot_action_dim = 7

class HumanRobotCoupledInterACT(nn.Module):
    """
    Coupled model for human-robot collaboration using InterACT-style asymmetric attention.
   
    Key features:
    - Human behavior: Graph structure encoded via GCN (ManiCast)
    - Robot behavior: Single 7-DOF arm with proprioceptive state
    - Shared visual observations from cameras
    - CLS tokens enable asymmetric attention for implicit environment modeling
    - Separate action predictions for human and robot
    """
   
    def __init__(self, config):
        """
        Args:
            config: Configuration object with the following attributes:
                # Human GCN parameters
                - human_input_channels: Number of channels for human joint coordinates (default: 3)
                - human_input_time_frame: Input sequence length for human
                - human_output_time_frame: Output sequence length for human
                - human_joints: Number of human joints to consider
                - human_action_dim: Dimension of human action output
               
                # Robot parameters
                - robot_state_dim: Robot proprioceptive state dimension (default: 7)
                - robot_action_dim: Robot action dimension (default: 7)
               
                # Vision parameters
                - image_shapes: Dict of camera observations, e.g., {"top": [3, 480, 640]}
                - vision_backbone: Vision encoder backbone (default: "resnet18")
                - pretrained_backbone_weights: Pretrained weights for backbone
               
                # Transformer parameters
                - dim_model: Hidden dimension (default: 512)
                - n_heads: Number of attention heads (default: 8)
                - dim_feedforward: Feedforward dimension (default: 3200)
                - feedforward_activation: Activation function (default: "relu")
                - num_blocks: Number of encoder blocks (default: 3)
                - num_cls_tokens_human: CLS tokens for human (default: 3)
                - num_cls_tokens_robot: CLS tokens for robot (default: 3)
                - num_cls_tokens_image: CLS tokens for images (default: 3)
                - n_decoder_layers: Number of decoder layers (default: 4)
                - dropout: Dropout rate (default: 0.1)
                - pre_norm: Use pre-normalization (default: False)
               
                # Training parameters
                - chunk_size: Action prediction chunk size (default: 100)
        """
        super().__init__()
        self.config = config
        self.use_image = "image" in config.input_shapes
        self.num_cls_tokens_image = config.num_cls_tokens_image
        # Human behavior encoder (GCN) 这部分应该移到外面去，使用他人的预训练模型
        self.human_encoder = STSGCN(
            input_channels=config.human_input_channels,
            input_time_frame=config.human_input_time_frame,
            output_time_frame=config.human_output_time_frame,
            st_gcnn_dropout=config.dropout,
            joints_to_consider=config.human_joints,
            n_txcnn_layers=3,
            txc_kernel_size=[3, 3],
            txc_dropout=config.dropout
        )
       
        # Get the output dimension from GCN
        # After GCN: (B, T, C, V) -> flatten C*V as feature dimension
        self.human_feature_dim = config.human_input_channels * config.human_joints
       
        # CLS tokens
        self.num_cls_tokens_human = config.num_cls_tokens_human
        num_cls_tokens_human = config.num_cls_tokens_human
        self.cls_input_human = nn.Embedding(1, config.dim_model)
        cls_input_human = self.cls_input_human.weight
        self.cls_input_human = cls_input_human.repeat(num_cls_tokens_human, 1)

        self.num_cls_tokens_robot = config.num_cls_tokens_robot
        num_cls_tokens_robot = config.num_cls_tokens_robot
        self.cls_input_robot = nn.Embedding(1, config.dim_model)
        num_robot_input_token_encoder = num_cls_tokens_robot + 7  # 7 DOF robot state
        cls_input_robot = self.cls_input_robot.weight
        self.cls_input_robot = cls_input_robot.repeat(num_cls_tokens_robot, 1)  # (num_cls_robot, 1, dim_model)
        
        # Positional encodings for different modalities
        self.register_buffer(
            "human_encoder_pos_enc",
            create_sinusoidal_pos_embedding(num_cls_tokens_human, config.dim_model).unsqueeze(0)
        ) # human的位置编码第一维度应该与human的时间步数有关，这里只是cls token的位置编码，需要修改
        self.register_buffer(
            "robot_encoder_pos_enc",
            create_sinusoidal_pos_embedding(num_robot_input_token_encoder, config.dim_model).unsqueeze(0)
        )

        if self.use_image:
            num_cls_tokens_image = config.num_cls_tokens_image
            self.cls_input_image = nn.Embedding(1, config.dim_model)
            cls_input_image = self.cls_input_image.weight
            self.cls_input_image = cls_input_image.repeat(num_cls_tokens_image, 1)  # (num_cls_image, 1, dim_model)
          
            self.register_buffer(
                "image_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_cls_tokens_image, config.dim_model).unsqueeze(0)
            )

        self.register_buffer(
            "cls_encoder_pos_enc",
            create_sinusoidal_pos_embedding(num_cls_tokens_human + num_cls_tokens_robot + num_cls_tokens_image, config.dim_model).unsqueeze(0)
        )

        # Coupled encoder (InterACT-style with asymmetric attention)
        self.coupled_encoder = CoupledInterACTEncoder(config)
        self.coupled_decoder = CoupledInterACTDecoder(config)
       
        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # 图像2d位置编码
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        # Action heads
        self.action_head_human = nn.Linear(config.dim_model, config.human_action_dim)
        self.action_head_robot = nn.Linear(config.dim_model, 7)
       
        self._reset_parameters()
   
    def _reset_parameters(self):
        """Xavier-uniform initialization of transformer parameters."""
        from itertools import chain
        for p in chain(
            self.coupled_encoder.parameters(),
            self.coupled_decoder.parameters()
        ):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize new action heads if they exist
        if hasattr(self, 'action_head_human'):
            nn.init.xavier_uniform_(self.action_head_human.weight)
            nn.init.zeros_(self.action_head_human.bias)
        if hasattr(self, 'action_head_robot'):
            nn.init.xavier_uniform_(self.action_head_robot.weight)
            nn.init.zeros_(self.action_head_robot.bias)
   
    def forward(self, batch: dict[str, Tensor]):
        """
        Forward pass for training or inference.
       
        Args:
            batch: Dictionary containing:
                - "human_graph": (B, C, T, V) human skeleton graph sequence
                - "robot_state": (B, 7) robot proprioceptive state
                - "observation.images" (optional): Dict of camera images
                - "human_action" (optional, training): (B, chunk_size, human_action_dim)
                - "robot_action" (optional, training): (B, chunk_size, robot_action_dim)
       
        Returns:
            Dictionary with:
                - "human_action_pred": (B, chunk_size, human_action_dim)
                - "robot_action_pred": (B, chunk_size, robot_action_dim)
                - "loss" (if training): Combined loss
        """
        batch_size = batch["robot_state"].shape[0]

        encoder_in_pos_embed = list(self.human_encoder_pos_enc)
        encoder_in_pos_embed.extend(list(self.robot_encoder_pos_enc))
        
        cls_token_human = self.cls_token_human.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_cls_human, dim_model)
        cls_token_robot = self.cls_token_robot.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_cls_robot, dim_model)

        encoder_in_tokens = []

        # Flatten and project human features
        # (B, C, T, V) -> (B, T, C*V) -> (B, T, dim_model)
        human_state = batch["human_graph"]  # (B, C, T, V) Encode human behavior through GCN
        human_features = einops.rearrange(human_state, 'b c t v -> b t (c v)')
       
        # Encode robot state
        robot_state = batch["robot_state"]

        encoder_in_tokens.append(torch.cat([cls_token_human, human_features], dim=1))  # (B, num_cls_human + T, dim_model)
        encoder_in_tokens.append(torch.cat([cls_token_robot, robot_state], dim=1))  # (B, num_cls_robot + 7, dim_model)

        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []
            cls_token_image = self.cls_token_image.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_cls_image, dim_model)

            cam_pos_embed = self.encoder_cam_feat_pos_embed(batch["images"]).to(dtype=batch["images"].dtype)  # (B, dim_model, H, W)
            cam_features = batch["images"]  # (B, dim_model, H, W)
            all_cam_features.append(cam_features)
            all_cam_pos_embeds.append(cam_pos_embed)

            all_cam_features = torch.cat(all_cam_features, dim=-1)
            encoder_in_tokens.append(
                torch.cat([cls_token_image, einops.rearrange(all_cam_features, 'b c h w -> b (h w) c')], dim=1)
            )
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, dim=-1)
            encoder_in_pos_embed.append(
                torch.cat([list(self.image_encoder_pos_enc)[0], list(einops.rearrange(all_cam_pos_embeds, 'b c h w -> b (h w) c'))[0]], dim=0)
            )
       
        encoder_in_tokens = torch.cat(encoder_in_tokens, dim=1)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, dim=1)
        encoder_in_cls_pos_embed = torch.cat(list(self.cls_encoder_pos_enc))

        encoder_in_pos_embed = encoder_in_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)
        encoder_in_cls_pos_embed = encoder_in_cls_pos_embed.unsqueeze(1).expand(-1, encoder_in_tokens.size(0), -1)
       
        # Forward through coupled encoder
        encoder_out = self.coupled_encoder(
            encoder_in_tokens,
            pos_embed=encoder_in_pos_embed,
            pos_embed_cls=encoder_in_cls_pos_embed
        )
        
        encoder_out_human_full = encoder_out[:self.num_cls_tokens_human + self.human_output_time_frame]
        encoder_in_pos_embed_human_full = encoder_in_pos_embed[:self.num_cls_tokens_human + self.human_output_time_frame]
        encoder_out_human_cls = encoder_out[:self.num_cls_tokens_human]
        encoder_in_pos_embed_human_cls = encoder_in_pos_embed[:self.num_cls_tokens_human]
        
        robot_start_idx = self.num_cls_tokens_human + self.human_output_time_frame
        encoder_out_robot_full = encoder_out[robot_start_idx: robot_start_idx + self.num_cls_tokens_robot + 7]
        encoder_in_pos_embed_robot_full = encoder_in_pos_embed[robot_start_idx: robot_start_idx + self.num_cls_tokens_robot + 7]
        encoder_out_robot_cls = encoder_out[robot_start_idx: robot_start_idx + self.num_cls_tokens_robot]
        encoder_in_pos_embed_robot_cls = encoder_in_pos_embed[robot_start_idx: robot_start_idx + self.num_cls_tokens_robot]

        encoder_out_image = encoder_out[robot_start_idx + self.num_cls_tokens_robot + 7:]
        encoder_in_pos_embed_image = encoder_in_pos_embed[robot_start_idx + self.num_cls_tokens_robot + 7:]

        human_encoder_context = torch.cat([
            encoder_out_human_full,      # full human features segment (cls + features)
            encoder_out_robot_cls,       # only robot cls tokens
            encoder_out_image            # image features
        ], dim=0)

        human_encoder_pos = torch.cat([
            encoder_in_pos_embed_human_full,
            encoder_in_pos_embed_robot_cls,
            encoder_in_pos_embed_image
        ], dim=0)

        robot_encoder_context = torch.cat([
            encoder_out_human_cls,
            encoder_out_robot_full,
            encoder_out_image
        ], dim=0)

        robot_encoder_pos = torch.cat([
            encoder_in_pos_embed_human_cls,
            encoder_in_pos_embed_robot_full,
            encoder_in_pos_embed_image
        ], dim=0)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_out.dtype,
            device=encoder_out.device
        )

        human_output, robot_output = self.coupled_decoder(
            decoder_in,
            human_encoder_context,
            robot_encoder_context,
            human_encoder_pos,
            robot_encoder_pos,
            self.decoder_pos_embed.weight.unsqueeze(1)
        )

        human_output = human_output.transpose(0, 1)  # (B, chunk_size, dim_model)
        robot_output = robot_output.transpose(0, 1)  # (B, chunk

        if hasattr(self, 'action_head_human') and hasattr(self, 'action_head_robot'):
            human_actions = self.action_head_human(human_output)
            robot_actions = self.action_head_robot(robot_output)
            actions = torch.cat([human_actions, robot_actions], dim=-1)
        else:
            raise ValueError("Action heads not defined in the model.")
    
        return actions


class ACTEncoderLayer(nn.Module):
    """Standard transformer encoder layer."""
   
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.dim_model,
            config.n_heads,
            dropout=config.dropout
        )
       
        # Feed forward layers
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
       
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
       
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
   
    def forward(self, x, pos_embed=None, key_padding_mask=None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class CoupledInterACTEncoder(nn.Module):
    """
    Coupled encoder implementing InterACT-style asymmetric attention mechanism.
    Enables implicit environment modeling and information exchange between human and robot.
    """
   
    def __init__(self, config):
        super().__init__()
       
        self.num_blocks = config.num_blocks
        self.num_cls_tokens_human = config.num_cls_tokens_human
        self.num_cls_tokens_robot = config.num_cls_tokens_robot
        self.num_cls_tokens_image = config.num_cls_tokens_image
       
        # Segment-wise encoders (within modality attention)
        self.segment_wise_encoder = nn.ModuleList([
            ACTEncoderLayer(config) for _ in range(config.num_blocks)
        ])
       
        # Cross-segment encoders (across modality attention)
        self.cross_segment_encoder = nn.ModuleList([
            ACTEncoderLayer(config) for _ in range(config.num_blocks)
        ])
   
    def forward(self, segments, pos_embed, pos_embed_cls):
        """
        Args:
            segments: (S, B, D) all tokens concatenated
            pos_embed: (S, B, D) positional embeddings for all tokens
            pos_embed_cls: (num_cls_total, B, D) positional embeddings for CLS tokens
       
        Returns:
            (S, B, D) encoded features with implicit environment coupling
        """
        segments = einops.rearrange(segments, 'b s d -> s b d')
        # Order: [human_cls, human_features, robot_cls, robot_features, image_cls (opt), image_features (opt)]
        segment_human = segments[:self.num_cls_tokens_human]
        segment_robot = segments[
            self.num_cls_tokens_human:
            self.num_cls_tokens_human + self.num_cls_tokens_robot + 7
        ]
        segment_image = segments[
            self.num_cls_tokens_human + self.num_cls_tokens_robot + 7:
        ]
        # Positional embeddings
        pos_embed_human = pos_embed[:self.num_cls_tokens_human]
        pos_embed_robot = pos_embed[
            self.num_cls_tokens_human:
            self.num_cls_tokens_human + self.num_cls_tokens_robot + 7
        ]
        pos_embed_image = pos_embed[
            self.num_cls_tokens_human + self.num_cls_tokens_robot + 7:
        ]

        # For simplicity in this implementation, we process all segments together
        # In a full implementation, you would split by modality
        for i in range(self.num_blocks):
            # Segment-wise attention (within modality)
            updated_segment_human = self.segment_wise_encoder[i](
                segment_human,
                pos_embed=pos_embed_human,
                key_padding_mask=None
            )
            updated_segment_robot = self.segment_wise_encoder[i](
                segment_robot,
                pos_embed=pos_embed_robot,
                key_padding_mask=None
            )
            updated_segment_image = self.segment_wise_encoder[i](
                segment_image,
                pos_embed=pos_embed_image,
                key_padding_mask=None
            )
           
            # Cross-segment attention (across modalities, CLS tokens mediate)
            updated_cls_tokens = self.cross_segment_encoder[i](
                torch.cat([
                    updated_segment_human[:self.num_cls_tokens_human],
                    updated_segment_robot[:self.num_cls_tokens_robot],
                    updated_segment_image[:self.num_cls_tokens_image]
                ], dim=0),
                pos_embed=pos_embed_cls,
                key_padding_mask=None
            )

            segment_human = torch.cat([
                updated_cls_tokens[:self.num_cls_tokens_human],
                updated_segment_human[self.num_cls_tokens_human:]
            ], dim=0)
            segment_robot = torch.cat([
                updated_cls_tokens[self.num_cls_tokens_human:
                                   self.num_cls_tokens_human + self.num_cls_tokens_robot],
                updated_segment_robot[self.num_cls_tokens_robot:]
            ], dim=0)
            segment_image = torch.cat([
                updated_cls_tokens[self.num_cls_tokens_human + self.num_cls_tokens_robot:],
                updated_segment_image[self.num_cls_tokens_image:]
            ], dim=0)

        # Concatenate all segments back
        segments = torch.cat([segment_human, segment_robot, segment_image], dim=0)

        return segments

class CoupledInterACTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.human_pre_decoder = nn.ModuleList([
            ACTDecoderLayer(config) for _ in range(config.n_pre_decoder_layers)
        ])
        self.robot_pre_decoder = nn.ModuleList([
            ACTDecoderLayer(config) for _ in range(config.n_pre_decoder_layers)
        ])

        self.sync_block = nn.ModuleList([
            nn.MultiheadAttention(
                config.dim_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=False
            ) for _ in range(config.n_sync_decoder_layers)
        ])

        # Post-synchronization layers
        self.human_post_decoder = nn.ModuleList([
            ACTDecoderLayer(config) for _ in range(config.n_post_decoder_layers)
        ])
        self.robot_post_decoder = nn.ModuleList([
            ACTDecoderLayer(config) for _ in range(config.n_post_decoder_layers)
        ])

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)

        self.num_cls_tokens_human = config.num_cls_tokens_human
        self.num_cls_tokens_robot = config.num_cls_tokens_robot
    
    def forward(
        self,
        decoder_input: Tensor,
        human_encoder_context: Tensor,
        robot_encoder_context: Tensor,
        human_encoder_pos: Tensor,
        robot_encoder_pos: Tensor,
        decoder_pos_embed: Tensor,
    ) -> tuple[Tensor, Tensor]:

        human_output = decoder_input.clone()
        robot_output = decoder_input.clone()

        for layer in self.human_pre_decoder:
            human_output = layer(
                human_output,
                human_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=human_encoder_pos
            )

        for layer in self.robot_pre_decoder:
            robot_output = layer(
                robot_output,
                robot_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=robot_encoder_pos
            )
        
        concatenated = torch.cat([human_output, robot_output], dim=0)

        for sync_layer in self.sync_block:
            concatenated_with_pos = concatenated + torch.cat([decoder_pos_embed, decoder_pos_embed], dim=0)
            synchronized, _ = sync_layer(
                concatenated_with_pos,
                concatenated_with_pos,
                concatenated
            )
            concatenated = concatenated + synchronized
        
        chunk_size = human_output.size(0)
        human_output = concatenated[:chunk_size]
        robot_output = concatenated[chunk_size:]

        for layer in self.human_post_decoder:
            human_output = layer(
                human_output,
                human_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=human_encoder_pos
            )
        for layer in self.robot_post_decoder:
            robot_output = layer(
                robot_output,
                robot_encoder_context,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=robot_encoder_pos
            )
        
        human_output = self.norm1(human_output)
        robot_output = self.norm2(robot_output)

        return human_output, robot_output


class ACTDecoder(nn.Module):
    def __init__(self, config):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """Standard transformer decoder layer with cross-attention."""
   
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
       
        # Feed forward
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
       
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
       
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
   
    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None, encoder_pos_embed: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for image features."""
   
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.temperature = 10000
        self.scale = 2 * np.pi
   
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image features
        Returns:
            (B, dimension*2, H, W) positional embeddings
        """
        B, C, H, W = x.shape
       
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
       
        # Normalize to [0, 2π]
        y_embed = (y_embed + 0.5) / H * self.scale
        x_embed = (x_embed + 0.5) / W * self.scale
       
        # Create sinusoidal embeddings
        dim_t = torch.arange(self.dimension, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dimension)
       
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
       
        # Apply sin/cos
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
       
        # Expand to spatial dimensions
        pos_x = pos_x[None, :, None, :].expand(B, -1, H, -1)  # (B, W, H, D)
        pos_y = pos_y[None, :, :, None].expand(B, -1, -1, W)  # (B, H, H, D)
       
        # Concatenate and rearrange
        pos = torch.cat([pos_y, pos_x], dim=1)  # (B, H, H, 2*D) - needs fixing
       
        # Actually do it correctly:
        pos_y = pos_y.permute(0, 3, 2, 1)  # (B, D, H, W)
        pos_x = pos_x.permute(0, 3, 2, 1)  # (B, D, H, W)
        pos = torch.cat([pos_y, pos_x], dim=1)  # (B, 2*D, H, W)
       
        return pos


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings."""
   
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (i // 2) / dimension)
            for i in range(dimension)
        ]
   
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def get_activation_fn(activation: str) -> Callable:
    """Return activation function given string name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


