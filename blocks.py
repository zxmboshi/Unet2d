import torch
import torch.nn as nn 
import torch.nn.functional as F


# shape_in: [batch_size, in_channels, vol_height, vol_width, vol_depth]
# shape_out: [batch_size, out_channels, vol_height, vol_width, vol_depth]
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        return self.norm(self.act(self.conv(x)))

# src: source img or vol needed to transform
# shape: [batch_size, 1, img_height, img_width] (2D)
# or: [batch_size, 1, vol_height, vol_width, vol_depth] (3D)
# flow: deformation field 
# shape: [batch_size, 2, img_height, img_width] (2D)
# or: [batch_size, 3, vol_height, vol_width, vol_depth] (3D)
class SpatialTransformerBlock(nn.Module):
    """
    N-D Spatial Transformer N = 2,3
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        # F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        # torch v1.2.0 is short of attribute 'align_corners' 
        return F.grid_sample(src, new_locs, mode=self.mode)


class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_size=4, num_heads=4, mode='down'):
        super().__init__()

        # patch embedded 
        emd_dim = dim * patch_size * patch_size * patch_size
        self.patch_emd_q = nn.Conv3d(dim, emd_dim, patch_size, patch_size)
        self.patch_emd_k = nn.Conv3d(dim, emd_dim, patch_size, patch_size)
        self.patch_emd_v = nn.Conv3d(dim, emd_dim, patch_size, patch_size) 

        # without position embedded 

        # attention 
        self.num_heads = num_heads
        head_dim = emd_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.patch_size = patch_size

        # out conv 
        self.conv_out = nn.Conv3d(dim, dim, 3, 1, 1)

        # same conv 
        self.conv_same = nn.Conv3d(dim, dim, 1, 1)
        self.mode = mode

        if self.mode == 'down':
            self.conv_down = nn.Conv3d(dim, dim, 1, 2)
        
        if self.mode == 'up':
            self.conv_up = nn.ConvTranspose3d(dim, dim, 2, 2)

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = int((H * W * D) // (self.patch_size ** 3))

        # patch embedded to get q,k,v 
        q, k, v = self.patch_emd_q(x), self.patch_emd_k(x), self.patch_emd_v(x)
        # .patch_emd() -> [batch_size, emd_dim, num_patches_h, num_patches_w, num_patches_d]
        q, k, v = q.flatten(2).transpose(-1, -2), k.flatten(2).transpose(-1, -2), v.flatten(2).transpose(-1, -2)
        # .flatten(2) -> [batch_size, emd_dim, num_patches]
        # .transpose(-1, -2) -> [batch_size, num_patches, emd_dim]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # .reshape -> [batch_size, num_patches, num_heads, emd_dim_per_head]
        # permute -> [batch_size, num_heads, num_patches, emd_dim_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # .transpose -> [batch_size, num_heads, emd_dim_per_head, num_patches]
        # @ -> [batch_size, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1) # according to row average
        y = (attn @ v).transpose(1, 2).flatten(2).reshape(B, C, H, W, D)
        # @: -> [batch_size, num_heads, num_patches, emd_dim_per_head]
        # .transpose(1, 2) -> [batch_size, num_patches, num_heads, emd_dim_per_head]
        # .flatten(2) -> [batch_size, num_patches, emd_dim]
        # reshape() -> [batch_size, in_dim, feature_height, feature_width, feature_depth]
        y = y + x # skip connect 
        # out conv 
        y = self.conv_out(y)
        # same conv 
        y_same = self.conv_same(y)
        # up conv or down conv 
        if self.mode == 'up':
            y_opt = self.conv_up(y)
        if self.mode == 'down':
            y_opt = self.conv_down(y)
        return y_same, y_opt


class TransformerBlockSingle(nn.Module): 
    def __init__(self, dim, patch_size=4, num_heads=4):
        super().__init__()
        
        super().__init__()

        # patch embedded 
        emd_dim = dim * patch_size * patch_size * patch_size
        self.patch_emd_q = nn.Conv3d(dim, emd_dim, patch_size, patch_size)
        self.patch_emd_k = nn.Conv3d(dim, emd_dim, patch_size, patch_size)
        self.patch_emd_v = nn.Conv3d(dim, emd_dim, patch_size, patch_size) 

        # without position embedded 

        # attention 
        self.num_heads = num_heads
        head_dim = emd_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.patch_size = patch_size

        # out conv 
        self.conv_out = nn.Conv3d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = int((H * W * D) // (self.patch_size ** 3))

        # patch embedded to get q,k,v 
        q, k, v = self.patch_emd_q(x), self.patch_emd_k(x), self.patch_emd_v(x)
        # .patch_emd() -> [batch_size, emd_dim, num_patches_h, num_patches_w, num_patches_d]
        q, k, v = q.flatten(2).transpose(-1, -2), k.flatten(2).transpose(-1, -2), v.flatten(2).transpose(-1, -2)
        # .flatten(2) -> [batch_size, emd_dim, num_patches]
        # .transpose(-1, -2) -> [batch_size, num_patches, emd_dim]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # .reshape -> [batch_size, num_patches, num_heads, emd_dim_per_head]
        # permute -> [batch_size, num_heads, num_patches, emd_dim_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # .transpose -> [batch_size, num_heads, emd_dim_per_head, num_patches]
        # @ -> [batch_size, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1) # according to row average
        y = (attn @ v).transpose(1, 2).flatten(2).reshape(B, C, H, W, D)
        # @: -> [batch_size, num_heads, num_patches, emd_dim_per_head]
        # .transpose(1, 2) -> [batch_size, num_patches, num_heads, emd_dim_per_head]
        # .flatten(2) -> [batch_size, num_patches, emd_dim]
        # reshape() -> [batch_size, in_dim, feature_height, feature_width, feature_depth]
        y = y + x # skip connect 
        # out conv 
        y = self.conv_out(y)
        return y 