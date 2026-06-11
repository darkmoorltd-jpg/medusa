import torchvision.transforms as T
import monai.transforms as MT

def get_2d_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_3d_transforms(img_size=128):
    return MT.Compose([
        MT.Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
        MT.Resize((img_size, img_size, img_size)),
        MT.ScaleIntensity(),
        MT.ToTensor(dtype=torch.float32)
    ])