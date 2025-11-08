from torchvision import transforms


def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),  # Slightly larger resize
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Random rotation between -10 and 10 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # Random perspective
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Random erasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transform(image_size=224):
    return get_val_transform(image_size)
