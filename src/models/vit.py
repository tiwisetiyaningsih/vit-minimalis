
import math

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 28
CHANNELS = 1
PATCH_SIZE = 4

NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 49

EMBED_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 16

NUM_LAYERS = 2
MLP_DIM = 128
NUM_CLASSES = 10

DROPOUT = 0.1


# ============================================================
# 1. PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Membagi gambar menjadi patch 4x4,
    kemudian mengubah setiap patch menjadi embedding 64 dimensi.

    Input : [B, 1, 28, 28]
    Output: [B, 49, 64]
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, images):
        batch_size, channels, height, width = images.shape

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Ukuran gambar harus {self.image_size}x{self.image_size}, "
                f"ditemukan {height}x{width}."
            )

        if channels != CHANNELS:
            raise ValueError(
                f"Jumlah channel harus {CHANNELS}, ditemukan {channels}."
            )

        # Membagi gambar menjadi patch 4x4
        patches = images.unfold(
            2, self.patch_size, self.patch_size
        ).unfold(
            3, self.patch_size, self.patch_size
        )

        # [B, C, 7, 7, 4, 4] -> [B, 49, 16]
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        patches = patches.reshape(
            batch_size,
            self.num_patches,
            -1,
        )

        # [B, 49, 16] -> [B, 49, 64]
        embeddings = self.projection(patches)

        return embeddings


# ============================================================
# 2. MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Implementasi manual Scaled Dot-Product Attention.

    Tidak menggunakan nn.MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension harus habis dibagi jumlah heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.attention_dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape

        # Menghasilkan Query, Key, dan Value
        qkv = self.qkv(x)

        # [B, N, 3*D] -> [B, N, 3, H, Head_Dim]
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        # [3, B, H, N, Head_Dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) / self.scale

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        # Menggabungkan seluruh attention heads
        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            embed_dim,
        )

        output = self.output_projection(attention_output)

        output = self.projection_dropout(output)

        return output


# ============================================================
# 3. TRANSFORMER ENCODER BLOCK
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder dengan:
    - Pre-LayerNorm
    - Multi-Head Self-Attention
    - Residual Connection
    - Feed Forward Network
    - Residual Connection
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention + Residual Connection
        x = x + self.attention(self.norm1(x))

        # Feed Forward Network + Residual Connection
        x = x + self.mlp(self.norm2(x))

        return x


# ============================================================
# 4. VISION TRANSFORMER
# ============================================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer minimalis untuk Fashion-MNIST.

    Input : [B, 1, 28, 28]
    Output: [B, 10]
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        channels=CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # Learnable Class Token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Learnable Positional Embedding: 49 patch + 1 CLS
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.pos_dropout = nn.Dropout(dropout)

        # Transformer Encoder sebanyak 2 blok
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def prepare_tokens(self, images):
        """
        Patch Embedding + Class Token + Positional Embedding.

        Output: [B, 50, 64]
        """

        batch_size = images.shape[0]

        # [B, 49, 64]
        patches = self.patch_embed(images)

        # [B, 1, 64]
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        # [B, 50, 64]
        tokens = torch.cat(
            (cls_tokens, patches),
            dim=1,
        )

        # Menambahkan positional embedding
        tokens = tokens + self.pos_embed

        tokens = self.pos_dropout(tokens)

        return tokens

    def forward(self, images):
        # Patch + CLS + Positional Embedding
        x = self.prepare_tokens(images)

        # Transformer Encoder
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Mengambil representasi Class Token
        cls_output = x[:, 0]

        # Classification Head
        logits = self.head(cls_output)

        return logits


# ============================================================
# 5. FORWARD PASS TEST
# ============================================================

if __name__ == "__main__":
    from src.data.dataset import create_dataloaders

    device = torch.device("cpu")

    print("=" * 60)
    print("VISION TRANSFORMER - FORWARD PASS TEST")
    print("=" * 60)

    # Membuat model
    model = VisionTransformer().to(device)

    print("\n[MODEL CONFIGURATION]")
    print(f"Image size       : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patch size       : {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Number of patches: {NUM_PATCHES}")
    print(f"Embedding dim    : {EMBED_DIM}")
    print(f"Number of heads  : {NUM_HEADS}")
    print(f"Head dimension   : {HEAD_DIM}")
    print(f"Encoder blocks   : {NUM_LAYERS}")
    print(f"MLP              : {EMBED_DIM} -> {MLP_DIM} -> {EMBED_DIM}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Mengambil satu batch dari DataLoader
    train_loader, _ = create_dataloaders()

    images, labels = next(iter(train_loader))

    images = images.to(device)

    print("\n[FORWARD PASS]")

    model.eval()

    with torch.no_grad():
        patch_embeddings = model.patch_embed(images)

        tokens = model.prepare_tokens(images)

        outputs = model(images)

    print(f"Input shape           : {images.shape}")
    print(f"Patch embedding shape : {patch_embeddings.shape}")
    print(f"Tokens shape          : {tokens.shape}")
    print(f"Output shape          : {outputs.shape}")

    print("\n[MODEL VALIDATION]")

    assert patch_embeddings.shape == (64, 49, 64)
    assert tokens.shape == (64, 50, 64)
    assert outputs.shape == (64, 10)

    assert torch.isfinite(patch_embeddings).all()
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(outputs).all()

    assert outputs.device.type == "cpu"

    print("Patch embedding : PASSED")
    print("Class token     : PASSED")
    print("Positional embed: PASSED")
    print("Transformer     : PASSED")
    print("Classification  : PASSED")
    print("NaN / Inf check : PASSED")
    print("CPU execution   : PASSED")

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("VIT FORWARD PASS TEST PASSED!")
    print("=" * 60)
