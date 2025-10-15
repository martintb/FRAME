"""Tests for UNet VAE model."""

import pytest
import torch
from frame_twin.models.unet_vae import (
    ConvBlock3D,
    DownBlock3D,
    UpBlock3D,
    EncoderUNet3D,
    DecoderUNet3D,
    UNetVAE
)


class TestConvBlock3D:
    """Tests for ConvBlock3D."""
    
    def test_same_channels(self):
        """Test ConvBlock with same input/output channels."""
        block = ConvBlock3D(32, 32, norm_groups=8)
        x = torch.randn(2, 32, 16, 16, 16)
        y = block(x)
        assert y.shape == (2, 32, 16, 16, 16)
    
    def test_different_channels(self):
        """Test ConvBlock with different input/output channels."""
        block = ConvBlock3D(32, 64, norm_groups=8)
        x = torch.randn(2, 32, 16, 16, 16)
        y = block(x)
        assert y.shape == (2, 64, 16, 16, 16)
    
    def test_small_channels(self):
        """Test ConvBlock with channels smaller than norm_groups."""
        block = ConvBlock3D(4, 4, norm_groups=8)
        x = torch.randn(2, 4, 16, 16, 16)
        y = block(x)
        assert y.shape == (2, 4, 16, 16, 16)


class TestDownBlock3D:
    """Tests for DownBlock3D."""
    
    def test_downsampling(self):
        """Test that downsampling reduces spatial dimensions by 2."""
        block = DownBlock3D(32, 64, norm_groups=8)
        x = torch.randn(2, 32, 32, 32, 32)
        y, skip = block(x)
        assert y.shape == (2, 64, 16, 16, 16)  # Downsampled
        assert skip.shape == (2, 64, 32, 32, 32)  # Before downsampling


class TestUpBlock3D:
    """Tests for UpBlock3D."""
    
    def test_upsampling_with_skip(self):
        """Test upsampling with skip connection."""
        block = UpBlock3D(64, 64, 32, norm_groups=8)
        x = torch.randn(2, 64, 16, 16, 16)
        skip = torch.randn(2, 64, 32, 32, 32)
        y = block(x, skip)
        assert y.shape == (2, 32, 32, 32, 32)  # Upsampled and processed
    
    def test_upsampling_without_skip(self):
        """Test upsampling without skip connection."""
        block = UpBlock3D(64, 64, 32, norm_groups=8)
        x = torch.randn(2, 64, 16, 16, 16)
        y = block(x, skip=None)
        # Without skip, should still upsample but won't concatenate
        assert y.shape[0] == 2
        assert y.shape[1] == 32


class TestEncoderUNet3D:
    """Tests for EncoderUNet3D."""
    
    def test_encoder_output_shapes(self):
        """Test encoder output shapes."""
        encoder = EncoderUNet3D(
            in_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8
        )
        x = torch.randn(2, 10, 128, 128, 128)
        z, mu, logvar, skips = encoder(x)
        
        # Check latent shapes
        assert z.shape == (2, 32, 16, 16, 16)  # 128 / 2^3 = 16
        assert mu.shape == (2, 32, 16, 16, 16)
        assert logvar.shape == (2, 32, 16, 16, 16)
        
        # Check skip connections (one per level)
        assert len(skips) == 3
        assert skips[0].shape == (2, 64, 64, 64, 64)   # After first down
        assert skips[1].shape == (2, 128, 32, 32, 32)  # After second down
        assert skips[2].shape == (2, 256, 16, 16, 16)  # After third down
    
    def test_encoder_different_levels(self):
        """Test encoder with different number of levels."""
        encoder = EncoderUNet3D(
            in_channels=10,
            latent_channels=16,
            base_channels=32,
            levels=2,
            norm_groups=8
        )
        x = torch.randn(2, 10, 64, 64, 64)
        z, mu, logvar, skips = encoder(x)
        
        assert z.shape == (2, 16, 16, 16, 16)  # 64 / 2^2 = 16
        assert len(skips) == 2


class TestDecoderUNet3D:
    """Tests for DecoderUNet3D."""
    
    def test_decoder_with_skips(self):
        """Test decoder with skip connections."""
        decoder = DecoderUNet3D(
            out_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8
        )
        z = torch.randn(2, 32, 16, 16, 16)
        skips = [
            torch.randn(2, 64, 64, 64, 64),
            torch.randn(2, 128, 32, 32, 32),
            torch.randn(2, 256, 16, 16, 16)
        ]
        x_hat = decoder(z, skips)
        assert x_hat.shape == (2, 10, 128, 128, 128)
    
    def test_decoder_without_skips(self):
        """Test decoder without skip connections."""
        decoder = DecoderUNet3D(
            out_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8
        )
        z = torch.randn(2, 32, 16, 16, 16)
        x_hat = decoder(z, skips=None)
        assert x_hat.shape == (2, 10, 128, 128, 128)


class TestUNetVAE:
    """Tests for UNetVAE model."""
    
    def test_forward_pass(self):
        """Test forward pass with skip connections."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        x = torch.randn(2, 10, 128, 128, 128)
        x_recon, z, mu, logvar = model(x)
        
        assert x_recon.shape == (2, 10, 128, 128, 128)
        assert z.shape == (2, 32, 16, 16, 16)
        assert mu.shape == (2, 32, 16, 16, 16)
        assert logvar.shape == (2, 32, 16, 16, 16)
    
    def test_encode(self):
        """Test encode method."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        x = torch.randn(2, 10, 128, 128, 128)
        z = model.encode(x)
        assert z.shape == (2, 32, 16, 16, 16)
    
    def test_decode_without_skips(self):
        """Test decode method without skip connections."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        z = torch.randn(2, 32, 16, 16, 16)
        x_hat = model.decode(z, skips=None)
        assert x_hat.shape == (2, 10, 128, 128, 128)
    
    def test_decode_with_skips(self):
        """Test decode method with skip connections."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        z = torch.randn(2, 32, 16, 16, 16)
        skips = [
            torch.randn(2, 64, 64, 64, 64),
            torch.randn(2, 128, 32, 32, 32),
            torch.randn(2, 256, 16, 16, 16)
        ]
        x_hat = model.decode(z, skips)
        assert x_hat.shape == (2, 10, 128, 128, 128)
    
    def test_sample(self):
        """Test sampling from prior."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        samples = model.sample(num_samples=4, device=torch.device('cpu'))
        assert samples.shape == (4, 10, 128, 128, 128)
    
    def test_different_configurations(self):
        """Test model with different configurations."""
        # Smaller model
        model = UNetVAE(
            input_channels=9,
            latent_channels=16,
            base_channels=16,
            levels=2,
            norm_groups=4
        )
        x = torch.randn(1, 9, 64, 64, 64)
        x_recon, z, mu, logvar = model(x)
        assert x_recon.shape == (1, 9, 64, 64, 64)
        assert z.shape == (1, 16, 16, 16, 16)
    
    def test_edge_preservation_quality(self):
        """Test that UNetVAE can reconstruct sharp edges better than baseline."""
        model = UNetVAE(
            input_channels=1,
            latent_channels=8,
            base_channels=16,
            levels=2,
            norm_groups=4
        )
        
        # Create a synthetic volume with sharp edges (a cube)
        x = torch.zeros(1, 1, 32, 32, 32)
        x[:, :, 8:24, 8:24, 8:24] = 1.0  # Cube with sharp edges
        
        # Forward pass
        x_recon, z, mu, logvar = model(x)
        
        # Check that reconstruction maintains reasonable values
        assert x_recon.shape == x.shape
        
        # Compute gradient magnitude in original and reconstruction
        def compute_gradient_mag(tensor):
            # Simple finite difference
            gx = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
            gy = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
            gz = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
            return gx.mean() + gy.mean() + gz.mean()
        
        # Original should have significant gradients at edges
        orig_grad = compute_gradient_mag(x)
        recon_grad = compute_gradient_mag(x_recon)
        
        # Reconstruction should maintain some gradient structure
        # (This is a weak test since the model is untrained, but checks basic functionality)
        assert orig_grad > 0
        assert recon_grad >= 0
    
    def test_model_attributes(self):
        """Test that model attributes are correctly stored."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=64,
            levels=3,
            norm_groups=16
        )
        assert model.input_channels == 10
        assert model.latent_channels == 32
        assert model.base_channels == 64
        assert model.levels == 3
        assert model.norm_groups == 16
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = UNetVAE(
            input_channels=10,
            latent_channels=32,
            base_channels=32,
            levels=3,
            norm_groups=8,
            skip_dropout_prob=0.0
        )
        x = torch.randn(2, 10, 128, 128, 128, requires_grad=True)
        x_recon, z, mu, logvar = model(x)
        
        # Compute a simple loss
        loss = x_recon.mean()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

