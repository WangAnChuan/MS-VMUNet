import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class KvasirDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=256, is_train=True):
        super().__init__()
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.img_size = img_size
        self.is_train = is_train

        all_names = sorted(os.listdir(self.img_dir))
        # Fixed split with seed 42: 800 train, 100 val, 100 test
        rng = random.Random(42)
        indices = list(range(len(all_names)))
        rng.shuffle(indices)

        train_idx = indices[:800]
        val_idx = indices[800:900]
        test_idx = indices[900:1000]

        if split == "train":
            selected_idx = train_idx
        elif split == "val":
            selected_idx = val_idx
        elif split == "test":
            selected_idx = test_idx
        else:
            raise ValueError(f"Invalid split {split}")

        self.file_names = [all_names[i] for i in selected_idx]
        print(f"[Kvasir-SEG] Loaded {len(self.file_names)} samples for {split} set.")

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            "image": image,
            "mask": mask_tensor,
            "name": file_name
        }

def get_dataloaders(data_root=None, data_dir=None, batch_size=16, num_workers=4, img_size=256):
    target_dir = data_dir if data_dir is not None else data_root
    if target_dir is None:
        candidates = [
            "data/Kvasir-SEG",
            os.path.join(os.path.dirname(__file__), "data", "Kvasir-SEG"),
            "/root/autodl-tmp/polyp_mamba_project/data/Kvasir-SEG",
            "../datasets/kvasir-seg/Kvasir-SEG",
            "./Kvasir-SEG",
        ]
        for c in candidates:
            if os.path.isdir(c):
                target_dir = c
                break
        if target_dir is None:
            target_dir = "data/Kvasir-SEG"

    train_dataset = KvasirDataset(target_dir, split="train", img_size=img_size, is_train=True)
    val_dataset = KvasirDataset(target_dir, split="val", img_size=img_size, is_train=False)
    test_dataset = KvasirDataset(target_dir, split="test", img_size=img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
