import torch
import torch.nn as nn

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from time import time
from timm.models.layers import trunc_normal_, DropPath

from tools import utils

from .utils import LayerNorm, GRN
from .convnextv1_sparse import SparseConvNeXtV1
from .convnextv2_sparse import SparseConvNeXtV2
from .resnetv2_sparse import SparseResNet, BasicBlock, Bottleneck


class Decoder_Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class VideoMAC(nn.Module):
    """ Video Masked Autoencoder Meets ConvNets
    """
    def __init__(
                self,
                img_size=224,
                in_chans=3,
                depths=[3, 3, 27, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=256,
                patch_size=32,
                mask_ratio=0.6,
                mode='ConvNeXtV2',
                norm_pix_loss=False,
                compute_loss=True):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.compute_loss = compute_loss
        
        if mode == 'ConvNeXtV2':
            # encoder
            self.encoder = SparseConvNeXtV2(
                in_chans=in_chans, depths=depths, dims=dims, D=2, patch_size=patch_size)
            # decoder
            self.proj = nn.Conv2d(
                in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1)
        elif mode == 'ConvNeXtV1':
            # encoder
            self.encoder = SparseConvNeXtV1(
                in_chans=in_chans, depths=depths, dims=dims, D=2, patch_size=patch_size)
            # decoder
            self.proj = nn.Conv2d(
                in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1)
        elif mode == 'ResNet18':
            # encoder
            self.encoder = SparseResNet(BasicBlock, [2, 2, 2, 2])
            # decoder
            self.proj = nn.Conv2d(
                in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1)
        elif mode == 'ResNet50':
            # encoder
            self.encoder = SparseResNet(Bottleneck, [3, 4, 6, 3])
            # decoder
            self.proj = nn.Conv2d(
                in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1)

        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Decoder_Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)
        if mode == 'ConvNeXtV2' or mode == 'ConvNeXtV1':
            self.apply(self._init_weights_convnextv2)
        elif mode == 'ResNet18' or mode == 'ResNet50':
            self.apply(self._init_weights_resnet)

    def _init_weights_convnextv2(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def _init_weights_resnet(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            # nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            # nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio, ids_shuffle=None, ids_restore=None):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None or ids_restore is None:
            noise = torch.randn(N, L, device=x.device)
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_shuffle, ids_restore

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio, ids_shuffle=None, ids_restore=None):
        # generate random masks
        mask, ids_shuffle, ids_restore = self.gen_random_mask(imgs, mask_ratio, ids_shuffle, ids_restore)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask, ids_shuffle, ids_restore

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        # n, c, h, w = x.shape
        mask = self.upsample_mask(mask, int((x.shape[2] / (mask.shape[1] ** .5)))).unsqueeze(1).type_as(x)
        # mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, frames, ids_shuffle=None, ids_restore=None):
        # Bx3xHxW
        x, mask, ids_shuffle, ids_restore = self.forward_encoder(frames, self.mask_ratio, ids_shuffle, ids_restore)
        pred = self.forward_decoder(x, mask)
        loss_intra = self.forward_loss(frames, pred, mask)
        if self.compute_loss:
            return loss_intra, pred, mask, ids_shuffle, ids_restore
        else:
            return loss_intra, pred

def mac_tiny(**kwargs):
    model = VideoMAC(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def mac_small(**kwargs):
    model = VideoMAC(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def mac_base(**kwargs):
    model = VideoMAC(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def mac_small_isotropic(**kwargs):
    model = VideoMAC(
        depths=18, dims=384, **kwargs)
    return model

def mac_base_isotropic(**kwargs):
    model = VideoMAC(
        depths=18, dims=768, **kwargs)
    return model

def mac_r18(**kwargs):
    model = VideoMAC(
        depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], mode='ResNet18', **kwargs)
    return model

def mac_r50(**kwargs):
    model = VideoMAC(
        depths=[3, 4, 6, 3], dims=[256, 512, 1024, 2048], mode='ResNet50', **kwargs)
    return model

def mac_cnxv1_tiny(**kwargs):
    model = VideoMAC(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], mode='ConvNeXtV1', **kwargs)
    return model

def mac_cnxv1_small(**kwargs):
    model = VideoMAC(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], mode='ConvNeXtV1', **kwargs)
    return model
