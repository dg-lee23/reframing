# Camera pose encoder from CameraCtrl (https://github.com/hehao13/CameraCtrl),
# with the same parameter names so CameraCtrl.ckpt loads with strict=True.
import math
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor


class _PositionalEncoding(nn.Module):
    # cameractrl/models/motion_module.py::PositionalEncoding
    def __init__(self, d_model, dropout=0., max_len=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalSelfAttention(Attention):
    # cameractrl/models/motion_module.py::TemporalSelfAttention
    def __init__(self, attention_mode=None, temporal_position_encoding=False,
                 temporal_position_encoding_max_len=32, rescale_output_factor=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal_Self"
        self.pos_encoder = _PositionalEncoding(
            kwargs["query_dim"], max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None
        self.rescale_output_factor = rescale_output_factor
        self.set_processor(AttnProcessor())

    def set_use_memory_efficient_attention_xformers(self, *args, **kwargs):
        pass

    def set_processor(self, processor):
        if isinstance(processor, AttnProcessor) or not hasattr(self, "processor"):
            super().set_processor(processor)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        out = self.processor(self, hidden_states, encoder_hidden_states=None, attention_mask=attention_mask)
        return out / self.rescale_output_factor


class TemporalTransformerBlock(nn.Module):
    # cameractrl/models/motion_module.py::TemporalTransformerBlock
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 attention_block_types=("Temporal_Self", "Temporal_Self",), dropout=0.0,
                 norm_num_groups=32, cross_attention_dim=768, activation_fn="geglu",
                 attention_bias=False, upcast_attention=False, temporal_position_encoding=False,
                 temporal_position_encoding_max_len=32, encoder_hidden_states_query=(False, False),
                 attention_activation_scale=1.0, attention_processor_kwargs: Dict = {},
                 rescale_output_factor=1.0):
        super().__init__()
        attention_blocks, norms = [], []
        self.attention_block_types = attention_block_types
        for block_name in attention_block_types:
            attention_blocks.append(
                TemporalSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=cross_attention_dim if block_name in ['Temporal_Cross', 'Temporal_Pose_Adaptor'] else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                attention_mask=attention_mask,
            ) + hidden_states
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        return hidden_states


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class CameraPoseEncoder(nn.Module):

    def __init__(self,
                 downscale_factor,
                 channels=[320, 640, 1280, 1280],
                 nums_rb=3,
                 cin=64,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 attention_block_types=("Temporal_Self", ),
                 temporal_position_encoding=False,
                 temporal_position_encoding_max_len=16,
                 rescale_output_factor=1.0):
        super(CameraPoseEncoder, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = TemporalTransformerBlock(dim=out_dim,
                                                                    num_attention_heads=temporal_attention_nhead,
                                                                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                                                                    attention_block_types=attention_block_types,
                                                                    dropout=0.0,
                                                                    cross_attention_dim=None,
                                                                    temporal_position_encoding=temporal_position_encoding,
                                                                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                                                                    rescale_output_factor=rescale_output_factor)
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x):
        # x: plucker embedding (b, 6, f, H, W) -> list of 4 features, each ((b f), c, h, w)
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            features.append(x)
        return features
