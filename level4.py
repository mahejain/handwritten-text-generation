import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 0. DEVICE
# ─────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, max_images=5000):
        self.image_paths = []
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        count = 0
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(root, file))
                    count += 1
                    if count >= max_images:
                        break
            if count >= max_images:
                break

        print(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('L')
            return self.transform(img)
        except Exception:
            return torch.zeros(1, 32, 32)

# ─────────────────────────────────────────
# 2. GENERATOR
# ─────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, 1, 32, 32)

# ─────────────────────────────────────────
# 3. DISCRIMINATOR
# ─────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        return self.model(x)

# ─────────────────────────────────────────
# 4. SAVE SAMPLES
# ─────────────────────────────────────────

def save_samples(generator, epoch, noise_dim, num_samples=8):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        fake_images = generator(noise).cpu().squeeze().numpy()

    fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, num_samples, figsize=(16, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fake_images[i], cmap='gray')
        ax.axis('off')

    plt.suptitle(f"Epoch {epoch}", fontsize=10)
    plt.tight_layout()
    output_path = f"output/level4_epoch_{epoch}.png"
    plt.savefig(output_path, dpi=100, facecolor='white')
    plt.close()
    print(f"Saved samples: {output_path}")
    generator.train()

# ─────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────

def train_gan(generator, discriminator, dataloader, epochs=50, noise_dim=100):
    loss_fn = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("Training GAN...")

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            opt_d.zero_grad()
            real_output = discriminator(batch)
            loss_real = loss_fn(real_output, real_labels)

            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            loss_fake = loss_fn(fake_output, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            output = discriminator(fake_images)
            loss_g = loss_fn(output, real_labels)
            loss_g.backward()
            opt_g.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
            save_samples(generator, epoch + 1, noise_dim)

    print("Training complete!")

# ─────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    NOISE_DIM = 100
    BATCH_SIZE = 64
    EPOCHS = 100

    dataset = HandwritingDataset('archive/words', max_images=5000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(noise_dim=NOISE_DIM).to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    train_gan(generator, discriminator, dataloader, epochs=EPOCHS, noise_dim=NOISE_DIM)

    torch.save(generator.state_dict(), "level4_generator.pth")
    torch.save(discriminator.state_dict(), "level4_discriminator.pth")
    print("Models saved!")