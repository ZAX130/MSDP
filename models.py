import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

from functional import modetqkrpb_cu

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
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

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
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

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = x + self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, c),
            ConvInsBlock(c, c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(c, 2 * c),
            ConvInsBlock(2 * c, 2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )


    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class MdTv2(nn.Module):
    def __init__(self, channel, head_dim, num_heads, kernel_size=3, qk_scale=1, use_rpb=True):
        super(MdTv2, self).__init__()

        self.peblock = ProjectionLayer(channel, dim=head_dim * num_heads)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        v = grid.reshape(self.kernel_size**3, 3)
        self.register_buffer('v', v)

        # self.dfi = RegConv(3 * num_heads)

    def forward(self, F, M):
        q, k = self.peblock(F), self.peblock(M)
        B, H, W, T, C = q.shape

        q = q.reshape(B, H, W, T, self.num_heads, C // self.num_heads).permute(0,4,1,2,3,5) * self.scale  #1,heads,H,W,T,dims
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # 1, C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.reshape(B, self.num_heads, C // self.num_heads, H+pd,W+pd,T+pd).permute(0, 1, 3, 4, 5, 2) # 1,heads,H+2,W+2,T+2,dims
        attn = modetqkrpb_cu(q,k,self.rpb)
        attn = attn.softmax(dim=-1)  # B h H W T num_tokens
        motion = (attn @ self.v)  # B x N x heads x 1 x 3
        motion = motion.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, H, W, T)

        return motion


class DoubleResConv(nn.Module):  # input shape: n, h, w, d, c
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = ConvResBlock(num_channels, num_channels)
        self.conv2 = ConvResBlock(num_channels, num_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class RegHead_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 ):
        super().__init__()

        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):

        x_out = self.reg_head(x_in)

        return x_out

class DecoderBlock_bottom(nn.Module):
    def __init__(self, channels, head_dim, num_heads):
        super(DecoderBlock_bottom, self).__init__()

        self.mdt = MdTv2(channels, head_dim, num_heads)
        self.res_convs = DoubleResConv(channels*2+num_heads*3)
        self.dfconv = ConvInsBlock(channels*2+num_heads*3, channels)
        self.reghead = RegHead_block(channels)


    def forward(self, F, M):
        motion = self.mdt(F, M)
        df = self.res_convs(torch.cat((F, M, motion), dim=1))
        df = self.dfconv(df)
        flow = self.reghead(df)
        return df, flow

class DecoderBlock(nn.Module):
    def __init__(self, channels,df_c, head_dim, num_heads):
        super(DecoderBlock, self).__init__()

        self.upsample = UpConvBlock(in_channels=df_c, out_channels=channels, kernel_size=2, stride=2, padding=0)
        self.mdt = MdTv2(channels, head_dim, num_heads)
        self.res_convs = DoubleResConv(channels*3+num_heads*3)
        self.dfconv = ConvInsBlock(channels*3+num_heads*3, channels)
        self.reghead = RegHead_block(channels)


    def forward(self, F, M, df):
        motion = self.mdt(F, M)
        df = self.res_convs(torch.cat((F, M, df, motion), dim=1))
        df = self.dfconv(df)
        flow = self.reghead(df)
        return df, flow

class MSDP(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_channel=1,
                 channels=8,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=1):
        super(MSDP, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)
        # self.encoder = ResEncoder(in_channel=in_channel, first_out_channel=2*c)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.decoder1 = DecoderBlock(c, 2*c, head_dim, num_heads[4])
        self.decoder2 = DecoderBlock(2 * c, 4*c, head_dim, num_heads[3])
        self.decoder3 = DecoderBlock(4 * c, 8*c, head_dim, num_heads[2])
        self.decoder4 = DecoderBlock(8 * c, 8*c, head_dim, num_heads[1])
        self.decoder5 = DecoderBlock_bottom(8 * c, head_dim, num_heads[0])

        self.transformer = nn.ModuleList()
        self.scale_factors = []
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.scale_factors.append(torch.FloatTensor([(s // 2**i -1)/(s // 2**(i+1) -1) for s in inshape]).view(1,3,1,1,1).cuda())

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        df, flow = self.decoder5(F5, M5)

        flow = self.upsample_trilin(self.scale_factors[3]*flow)
        M4 = self.transformer[3](M4, flow)
        df = self.decoder4.upsample(df)
        df, w = self.decoder4(F4, M4, df)

        flow = self.upsample_trilin(self.scale_factors[2]*(self.transformer[3](flow,w)+w))
        M3 = self.transformer[2](M3, flow)
        df = self.decoder3.upsample(df)
        df, w = self.decoder3(F3, M3, df)

        flow = self.upsample_trilin(self.scale_factors[1] *(self.transformer[2](flow,w)+w))
        M2 = self.transformer[1](M2, flow)
        df = self.decoder2.upsample(df)
        df, w = self.decoder2(F2, M2, df)

        flow = self.upsample_trilin(self.scale_factors[0] *(self.transformer[1](flow,w)+w))
        M1 = self.transformer[0](M1, flow)
        df = self.decoder1.upsample(df)
        _, w = self.decoder1(F1, M1, df)

        flow = self.transformer[0](flow,w)+w
        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':

    inshape = (1, 1,64, 64,64)
    torch.cuda.set_device(1)
    model = MSDP(inshape[2:]).cuda()
    # print(str(model))
    A = torch.ones(inshape).cuda()
    B = torch.ones(inshape).cuda()
    out, flow = model(A.cuda(), B.cuda())
    print(out.shape, flow.shape)