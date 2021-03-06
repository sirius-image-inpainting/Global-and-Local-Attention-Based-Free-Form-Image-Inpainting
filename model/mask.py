import numpy as np
import torch
import math
from PIL import Image, ImageDraw


## From DEEPFILL2
class brush_stroke_mask(torch.nn.Module):
    def __init__(self):
        super(brush_stroke_mask, self).__init__()
        self.min_num_vertex = 4
        self.max_num_vertex = 12
        self.mean_angle = 2 * math.pi / 5
        self.angle_range = 2 * math.pi / 15
        self.min_width = 12
        self.max_width = 40

    def generate_mask(self,H, W):
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)
            angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        # mask = np.reshape(mask, (1, 1, H, W))
        return mask
