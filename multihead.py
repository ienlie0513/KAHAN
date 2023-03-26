import torch
import torch.nn as nn

class ClipImgAtt(nn.Module):
    def __init__(self):
        super(ClipImgAtt, self).__init__()
        self.attn = nn.MultiheadAttention(512, num_heads=4)

    def forward(self, img_input, clip_ent_vec, clip_clm_vec, clip_lk):
        img_input = img_input.permute(1, 0, 2)
        clip_ent_vec = clip_ent_vec.permute(1, 0, 2)
        clip_clm_vec = clip_clm_vec.permute(1, 0, 2)

        attn_output, _ = self.attn(img_input, clip_ent_vec, clip_clm_vec)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output

img_input = torch.randn(16, 1, 512)
clip_ent_vec = torch.randn(16, 5, 512)
clip_clm_vec = torch.randn(16, 5, 512)
clip_lk = torch.randn(16, 1)

clip_ent_input = (clip_ent_vec, clip_clm_vec, clip_lk)

model = ClipImgAtt()
image_vec = model(img_input, *clip_ent_input)
print(image_vec.shape)