import os
import random
import cv2
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# --- Dataset ---
class GlareDataset(Dataset):
    def __init__(self, gt_dir, transform=None):
        self.gt_paths = sorted(glob(os.path.join(gt_dir, "*.*")))
        self.transform = transform

    def add_glare(self, gt_image):
        """
        gt_image: NumPy array (H, W, C), uint8
        Returns distorted image (HWC), mask (HW)
        """
        h, w, c = gt_image.shape
        distorted = gt_image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        num_glare = random.randint(1, 3)
        for _ in range(num_glare):
            max_radius = min(h, w) // 4
            min_radius = max(10, max_radius // 4)
            if max_radius <= min_radius:
                radius = min_radius
            else:
                radius = random.randint(min_radius, max_radius)

            cx = random.randint(radius, w - radius)
            cy = random.randint(radius, h - radius)
            intensity = random.randint(200, 255)

            cv2.circle(distorted, (cx, cy), radius, (intensity, intensity, intensity), -1)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

        distorted = cv2.GaussianBlur(distorted, (0, 0), sigmaX=5, sigmaY=5)
        return distorted, mask

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_pil = Image.open(self.gt_paths[idx]).convert("RGB")
        gt_np = np.array(gt_pil)
        distorted_np, mask_np = self.add_glare(gt_np)

        if self.transform:
            gt = self.transform(gt_np)
            distorted = self.transform(distorted_np)
            # mask: convert to float tensor (1, H, W)
            mask = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0
        else:
            gt = torch.from_numpy(gt_np).permute(2,0,1).float() / 255.0
            distorted = torch.from_numpy(distorted_np).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0

        return distorted, mask, gt

# --- Transform ---
transform = transforms.Compose([
    transforms.ToTensor()
])

# --- Collate with padding ---
def collate_with_padding(batch):
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)

    distorted_batch, mask_batch, gt_batch = [], [], []

    for distorted, mask, gt in batch:
        pad_h = max_h - distorted.shape[1]
        pad_w = max_w - distorted.shape[2]
        distorted_batch.append(torch.nn.functional.pad(distorted, (0, pad_w, 0, pad_h)))
        mask_batch.append(torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h)))
        gt_batch.append(torch.nn.functional.pad(gt, (0, pad_w, 0, pad_h)))

    return torch.stack(distorted_batch), torch.stack(mask_batch), torch.stack(gt_batch)

# --- Simple UNet-like model ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Training ---
def train_model(gt_dir, epochs=5, batch_size=2, lr=1e-3):
    dataset = GlareDataset(gt_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_with_padding)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0

        for distorted, mask, gt in pbar:
            distorted, gt = distorted.to(device), gt.to(device)
            
            optimizer.zero_grad()
            output = model(distorted)

            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss}")

        # ⬇️ Save checkpoint every epoch
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    # ⬇️ Save final trained model
    torch.save(model.state_dict(), "glare_removal_model.pth")
    print("Saved final model to glare_removal_model.pth")

    return model


# --- Usage ---
gt_directory = "C:/Users/allan/Downloads/glare_dataset/original"  # <-- set your GT directory here
trained_model = train_model(gt_directory, epochs=5, batch_size=2)
