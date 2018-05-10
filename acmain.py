import model
import argparse
import torch
import dataset_generator
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', required=True, help='path to folder with images')
    parser.add_argument('--data-file', required=True, help='path to images description file')
    parser.add_argument('--experiment', required=True, help='where to store samples and models')
    parser.add_argument('--img-size', type=int, default=64, help='image size')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--n-workers', type=int, default=2, help='number of workers to load dataset')
    parser.add_argument('--d-iter', type=int, default=5, help='discriminator iters per generator iter')
    parser.add_argument('--n-channels', type=int, default=3, help='number of channels')
    parser.add_argument('--n-gen-features', type=int, default=64, help='number of generator features')
    parser.add_argument('--n-disc-features', type=int, default=64, help='number of discriminator features')
    parser.add_argument('--dim-z', type=int, default=100, help='dimensionality of a latent vector')
    parser.add_argument('--n-iter', type=int, default=20, help='number of training epochs')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--net-gen', default='', help='path to generator')
    parser.add_argument('--net-disc', default='', help='path to discriminator')
    parser.add_argument('--gp-coef', type=float, default=10, help='GP coef')
    parser.add_argument('--ac-coef', type=float, default=0.2, help='AC coef')
    parser.add_argument('--classes', type=int, nargs='+', help='Classes labels to train')
    parser.add_argument('--frequency', type=int, default=500, help='How often to save models and generate images')
    args = parser.parse_args()

    # Creating experiment folder
    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)

    # Creating models
    signs_subset = args.classes
    print(signs_subset)
    n_classes = len(args.classes)

    if args.net_gen:
        gen = torch.load(args.net_gen)
        print('loaded generator')
    else:
        gen = model.BNGenerator(img_size=args.img_size, y_dim=n_classes, z_dim=args.dim_z,
                          n_channels=args.n_channels, n_features=args.n_gen_features)
        gen.apply(model.weights_init)

    if args.net_disc:
        disc = torch.load(args.net_disc)
        print('loaded discriminator')
    else:
        disc = model.ACDiscriminator(img_size=args.img_size, n_classes=n_classes, n_channels=args.n_channels,
                               n_features=args.n_disc_features)
        disc.apply(model.weights_init)

    if args.cuda:
        gen.cuda()
        disc.cuda()

    # Creating optimizers
    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

    # Creating dataloader
    dataset = dataset_generator.SignsDataset(args.data_file, args.data_folder, signs_subset=signs_subset,
                                             transform=transforms.Compose([
                                                 transforms.Scale(args.img_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.n_workers, drop_last=True)

    # Creating auxiliary tensors
    noise = torch.FloatTensor(args.batch_size, args.dim_z)
    fixed_noise = torch.FloatTensor(n_classes * 8, args.dim_z).normal_(0, 1)
    fixed_labels = np.repeat(np.arange(n_classes), 8)
    fixed_labels = torch.LongTensor(fixed_labels)
    eps = torch.FloatTensor(args.batch_size, 1)
    ones = torch.FloatTensor(args.batch_size).fill_(1.0)
    weights = dataset.get_weights()
    one_hot = torch.FloatTensor(args.batch_size, 8)
    if args.cuda:
        noise = noise.cuda()
        fixed_noise = fixed_noise.cuda()
        fixed_labels = fixed_labels.cuda()
        eps = eps.cuda()
        ones = ones.cuda()
        weights = weights.cuda()
        one_hot = one_hot.cuda()

    # Training loop
    ac_coef = args.ac_coef
    gp_coef = args.gp_coef
    gen_iteration = 0
    while gen_iteration < args.n_iter:
        data = iter(dataloader)
        i = 0
        while i < len(dataloader):
            j = 0
            while j < args.d_iter and i < len(dataloader):
                real_images, real_labels = next(data)
                noise.normal_(0, 1)
                if args.cuda:
                    real_images = real_images.cuda()
                    real_labels = real_labels.cuda()
                current_weights = Variable(weights[real_labels])
                fake_labels = real_labels.clone()
                fake_images = gen(Variable(noise), Variable(fake_labels))

                eps.uniform_()
                alpha = eps.unsqueeze(2).unsqueeze(3).expand(real_images.size())
                interpolate_images = alpha * real_images + (1 - alpha) * fake_images.data

                opt_disc.zero_grad()
                wass_fake, logits_fake = disc(fake_images)
                wass_real, logits_real = disc(Variable(real_images))

                wass_loss = (wass_fake - wass_real)
                ce_loss = F.cross_entropy(logits_real, Variable(real_labels), weight=weights)
                ones.resize_as_(wass_loss.data)
                grad_norms = disc.grad_norm(interpolate_images, ones)
                grad_loss = (grad_norms - 1) ** 2
                disc_loss = (torch.mean(wass_loss.squeeze() * current_weights) + ce_loss +
                             gp_coef * torch.mean(grad_loss * current_weights))

                opt_disc.zero_grad()
                disc_loss.backward()
                opt_disc.step()

                j += 1
                i += 1

            noise.normal_(0, 1)
            current_weights = Variable(weights[fake_labels])
            fake_images = gen(Variable(noise), Variable(fake_labels))

            opt_gen.zero_grad()
            wass_fake, logits_fake = disc(fake_images)
            ce_loss = F.cross_entropy(logits_fake, Variable(fake_labels), weight=weights)
            gen_loss = -torch.mean(wass_fake.squeeze() * current_weights) + ac_coef * ce_loss
            gen_loss.backward()
            opt_gen.step()
            gen_iteration += 1

            print('[{}/{}] disc_loss = {} \t gen_loss = {}'.format(gen_iteration, args.n_iter,
                                                                       disc_loss.cpu().data.numpy()[0],
                                                                       gen_loss.cpu().data.numpy()[0]))
            if gen_iteration % args.frequency == 0 or gen_iteration == 1:
                disc_path = os.path.join(args.experiment, 'disc_{}.pth'.format(gen_iteration))
                gen_path = os.path.join(args.experiment, 'gen_{}.pth'.format(gen_iteration))
                torch.save(disc, disc_path)
                torch.save(gen, gen_path)

                fake_data = gen(Variable(fixed_noise), Variable(fixed_labels)).cpu().data
                fake_data = fake_data.mul(0.5).add(0.5)

                fake_data_path = os.path.join(args.experiment, 'fake_samples_{}.png'.format(gen_iteration))
                vutils.save_image(fake_data, fake_data_path, nrow=n_classes)


