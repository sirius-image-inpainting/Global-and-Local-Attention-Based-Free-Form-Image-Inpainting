import torch
import torch.nn as nn
import torch.nn.functional as F


class MPGA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, mask):
        batch_size, c, w, h = queries.size()
        keys_proj = self.key_conv(keys).view(batch_size, c, w * h)
        queries_proj = self.query_conv(queries).view(batch_size, c, w * h).permute(0, 2, 1)  # b, c, w * h -> n, w * h, c
        attention_scores = torch.bmm(queries_proj, keys_proj)  # n, w * h, w * h

        _, _, w_init, h_init = mask.size()
        mask = F.interpolate(mask, scale_factor=w / w_init)  # b, 1, w, h
        reshaped_mask = mask.view(batch_size, 1, w * h)
        reshaped_mask = reshaped_mask.repeat(1, w * h, 1).permute(0, 2, 1)  # b, 1, w * h -> b, w * h, w * h

        pruned_attention_scores = attention_scores * reshaped_mask
        attention_weights = self.softmax(pruned_attention_scores).permute(0, 2, 1)  # b, w * h, w * h
        # normalized keys on the first dim

        values = self.value_conv(keys).view(batch_size, c, w * h)  # b, c, w * h
        attention_output = torch.bmm(values, attention_weights).view(batch_size, c, w, h)  # b, c, w, h

        return queries * mask + (1 - mask) * attention_output


class GlobalAttentionPatch(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttentionPatch, self).__init__()
        self.chanel_in = in_dim

        self.query_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax_channel = nn.Softmax(dim=-1)
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, x, y, m):
        '''
        Something
        '''
        feature_size = list(x.size())
        # Channel attention
        query_channel = self.query_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        key_channel = self.key_channel(y).view(feature_size[0], -1, feature_size[2] * feature_size[3]).permute(0, 2, 1)
        channel_correlation = torch.bmm(query_channel, key_channel)
        m_r = m.view(feature_size[0], -1, feature_size[2] * feature_size[3])
        channel_correlation = torch.bmm(channel_correlation, m_r)
        energy_channel = self.softmax_channel(channel_correlation)
        value_channel = self.value_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        attented_channel = (energy_channel * value_channel).view(feature_size[0], feature_size[1], feature_size[2], feature_size[3])
        # import ipdb; ipdb.set_trace()
        out = x * m + self.gamma * (1.0 - m) * attented_channel
        return out


class GLA(nn.Module):
    def __init__(self, in_dim, patch_size=3, propagate_size=3, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.in_dim = in_dim
        self.feature_attention = MPGA(in_dim)
        self.patch_attention = GlobalAttentionPatch(in_dim)

    def forward(self, foreground, same_number_of_args, mask, background="same"):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if background == "same":
            background = foreground.clone()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        background = background * (1 - mask)
        foreground = self.feature_attention(foreground, background, mask)
        background = F.pad(background,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride)\
                                     .unfold(3, self.patch_size,self.stride)\
                                    .contiguous().view(bz, nc, -1, self.patch_size, self.patch_size)

        mask_resized = mask.repeat(1, self.in_dim, 1, 1)
        mask_resized = F.pad(mask_resized,
                             [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2,
                              self.patch_size // 2])
        mask_kernels_all = mask_resized.unfold(2, self.patch_size, self.stride)\
                                       .unfold(3, self.patch_size, self.stride)\
                                       .contiguous()\
                                       .view(bz, nc, -1, self.patch_size, self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        mask_kernels_all = mask_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]

            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            mask_kernels = mask_kernels_all[i]
            conv_kernels = self.patch_attention(conv_kernels, conv_kernels, mask_kernels)
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
            #             print(conv_result.shape)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones(
                        [conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1,
                                       groups=conv_result.size(1))
            mm = (torch.mean(mask_kernels_all[i], dim=[1, 2, 3], keepdim=True) == 0.0).to(torch.float32)
            mm = mm.permute(1, 0, 2, 3).cuda()
            conv_result = conv_result * mm
            attention_scores = F.softmax(conv_result, dim=1)
            attention_scores = attention_scores * mm

            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch_size // 2)
            output_tensor.append(recovered_foreground)
        return torch.cat(output_tensor, dim=0)
