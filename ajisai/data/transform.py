from torchvision import transforms

class AjisaiTransform:
    """
    Dual-crop transform (inspired by AMDIM & SimCLR)
    Produces one global and one local view of the same image.
    """
    def __init__(self):
        self.global_crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.669, 0.691, 0.685],
                                 std=[0.203, 0.193, 0.264])
        ])
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.3, 0.6)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.669, 0.691, 0.685],
                                 std=[0.203, 0.193, 0.264])
        ])

    def __call__(self, img):
        return self.global_crop(img), self.local_crop(img)