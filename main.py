import time
import gc

from model.network import Generator, Discriminator
from DataModule.PlacesDataModule import PlacesDataset

from argparse import ArgumentParser, Namespace
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import SSIM


class Network:
    def __init__(self, args, from_pretrained="", generator_path=None, discriminator_path=None,
                                                    generator_optimizer_path=None, discriminator_optimizer_path=None):
        super().__init__()
        self.args = args
        self.relu = nn.ReLU()
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr,
                                                        betas=(args.b1, args.b2))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.lr,
                                                    betas=(args.b1, args.b2))
        if from_pretrained:
            if generator_path is None or discriminator_path is None or \
                generator_optimizer_path is None or discriminator_optimizer_path is None:
                raise ValueError("To train from pretrain provide all paths.")
            checkpoint = torch.load(from_pretrained)
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)

    def save_weights(self, path):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
        }, path)

    @staticmethod
    def from_pretrained(path):
        checkpoint = torch.load(path)
        generator = Generator().load_state_dict(checkpoint['generator'])
        discriminator = Discriminator().load_state_dict(checkpoint['discriminator'])
        return generator, discriminator

    def forward_step(self, batch):
        ground_truth, mask = batch['ground_truth'], batch['mask']
        ground_truth = ground_truth.cuda()
        mask = mask.cuda()
        input_image = (1 - mask) * ground_truth
        # import ipdb; ipdb.set_trace()
        course_image, refinement_image = self.generator(input_image, mask)

        course_image_inpainted = mask * course_image + (1 - mask) * ground_truth
        refinement_image_inpainted = mask * refinement_image + (1 - mask) * ground_truth

        fake_feature_volumes, real_feature_volumes = self.discriminator(refinement_image_inpainted.detach(),
                                                                        ground_truth)
        discriminator_lorentzian_loss = torch.mean(torch.log(1 + torch.abs(fake_feature_volumes - real_feature_volumes)))
        discriminator_hinge_loss = (torch.mean(self.relu(1 - (real_feature_volumes - torch.mean(fake_feature_volumes, dim=0))))
                                 +  torch.mean(self.relu(1 + (fake_feature_volumes - torch.mean(real_feature_volumes, dim=0))))) / 2

        l1_loss = self.l1_loss((1 - mask) * course_image, (1 - mask) * ground_truth) * self.args.coarse_l1_alpha \
                + self.l1_loss((1 - mask) * refinement_image, (1 - mask) * ground_truth)
        ssim_loss = 1 / 2 * (1 - self.ssim(course_image_inpainted, ground_truth)) \
                  + 1 / 2 * (1 - self.ssim(refinement_image_inpainted, ground_truth))

        fake_feature_volumes, real_feature_volumes = self.discriminator(refinement_image_inpainted, ground_truth)
        generator_lorentzian_loss = torch.mean(torch.log(1 + torch.abs(fake_feature_volumes - real_feature_volumes)))
        generator_hinge_loss = (torch.mean(self.relu(1 + (real_feature_volumes - torch.mean(fake_feature_volumes, dim=0)))) +
                                torch.mean(self.relu(1 - (fake_feature_volumes - torch.mean(real_feature_volumes, dim=0))))) / 2
        loss = self.args.l1_weight * l1_loss + ssim_loss * (1 - self.args.l1_weight)
        return course_image_inpainted, \
               refinement_image_inpainted, \
               {'discriminator_loss': discriminator_lorentzian_loss + discriminator_hinge_loss
                   , 'generator_loss': self.args.l1_loss_alpha * loss + generator_lorentzian_loss + generator_hinge_loss
                   , 'l1': loss}

    def backward_step(self, losses):
        losses['generator_loss'].backward(retain_graph=True)
        self.generator_optimizer.step()

        losses['discriminator_loss'].backward()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()


def main(args: Namespace) -> None:
    model = Network(args)
    model.to('cuda:0')

    train_dataset = PlacesDataset("train")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
    # val_dataset = PlacesDataset("val")
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    # test_dataset = PlacesDataset("test")
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    logs_path = os.path.join(args.checkpoints_dir, "logs")
    save_weights_path = os.path.join(args.checkpoints_dir, "weights")
    writer = SummaryWriter(log_dir=logs_path)

    def denorm(x):
        out = (x + 1) / 2  # [-1,1] -> [0,1]
        return out.clamp_(0, 1)

    start = time.time()
    for i, batch in enumerate(train_dataloader):
        course_image, refinement_image, losses = model.forward_step(batch)
        model.backward_step(losses)
        if i % args.verbose_every == 0:
            finish = time.time()
            print("iterations complete", i, "time took:", finish - start)
            start = finish
            for k, v in losses.items():
                writer.add_scalar(k, v.item(), i)
                print(k, v.item())
            ims = torch.cat([(batch['ground_truth'] * (1 - batch['mask'])).cpu(), course_image.cpu(), refinement_image.cpu(), batch['ground_truth'].cpu()], dim=3)
            writer.add_image('raw_masked_coarse_refine', denorm(ims[0]), i)
            del ims
        if i % args.save_weights_every == 0:
            model.save_weights(save_weights_path)
        #gc_time = time.time()
        del losses, course_image, refinement_image, batch
        gc.collect()
        torch.cuda.empty_cache()
        #print("delete overhead", time.time() - gc_time)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    # dataloader params
    parser.add_argument('--batch_size', type=int, default=2)
    # optimizer params
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=tuple, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    # objectives param
    parser.add_argument('--coarse_l1_alpha', type=float, default=1.2)
    parser.add_argument('--l1_weight', type=float, default=0.75)
    parser.add_argument('--l1_loss_alpha', type=float, default=1.2)
    # verbose info
    parser.add_argument('--verbose_every', type=int, default=200)
    parser.add_argument('--save_weights_every', type=int, default=500)
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    # experiment
    #parser.add_argument('--mask_ratio')
    return parser


if __name__ == '__main__':
    parser = parse_args()
    main(parser.parse_args())

# normalize - ToTensor => * 2 - 1
# masking -
# losses
# train loop
