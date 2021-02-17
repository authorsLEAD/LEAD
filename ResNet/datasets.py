import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        self.train = torch.utils.data.DataLoader(
            Dt(root=args.data_path, train=True, transform=transform, download=True),
            batch_size=args.dis_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        self.valid = torch.utils.data.DataLoader(
            Dt(root=args.data_path, train=False, transform=transform),
            batch_size=args.dis_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        self.test = self.valid
