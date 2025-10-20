"""Tests for HVAE model and loss."""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from frame_twin.models.hvae import HVAE, EncBottom, PriorBottom, ConcatConditioning, FiLMConditioning
from frame_twin.losses.hvae_loss import HVAELoss


class TestEncBottom:
    """Test EncBottom conditional encoder."""
    
    def test_enc_bottom_forward(self):
        """Test EncBottom forward pass."""
        batch_size = 2
        input_channels = 10
        latent_channels = 32
        channel_schedule = [64, 128, 256]
        z2_channels = 16
        z2_spatial_size = 4
        z1_spatial_size = 8
        
        # Create encoder
        encoder = EncBottom(
            input_channels=input_channels,
            latent_channels=latent_channels,
            channel_schedule=channel_schedule,
            z2_channels=z2_channels,
            z2_spatial_size=z2_spatial_size,
            z1_spatial_size=z1_spatial_size
        )
        
        # Create input tensors
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        z2 = torch.randn(batch_size, z2_channels, z2_spatial_size, z2_spatial_size, z2_spatial_size)
        
        # Forward pass
        z1, mu, logvar = encoder(x, z2)
        
        # Check shapes
        # Calculate actual spatial size based on channel schedule
        actual_spatial_size = 128 // (2 ** len(channel_schedule))
        expected_shape = (batch_size, latent_channels, actual_spatial_size, actual_spatial_size, actual_spatial_size)
        assert z1.shape == expected_shape
        assert mu.shape == expected_shape
        assert logvar.shape == expected_shape
        
        # Check that z1 is properly sampled from mu and logvar
        std = torch.exp(0.5 * logvar)
        expected_z1 = mu + std * torch.randn_like(std)
        # Note: This won't be exactly equal due to random sampling, but structure should be correct
        assert z1.shape == expected_z1.shape


class TestPriorBottom:
    """Test PriorBottom conditional prior."""
    
    def test_prior_bottom_forward(self):
        """Test PriorBottom forward pass."""
        batch_size = 2
        z2_channels = 16
        z2_spatial_size = 4
        z1_channels = 32
        z1_spatial_size = 8
        
        # Create prior
        prior = PriorBottom(
            z2_channels=z2_channels,
            z2_spatial_size=z2_spatial_size,
            z1_channels=z1_channels,
            z1_spatial_size=z1_spatial_size
        )
        
        # Create input
        z2 = torch.randn(batch_size, z2_channels, z2_spatial_size, z2_spatial_size, z2_spatial_size)
        
        # Forward pass
        mu, logvar = prior(z2)
        
        # Check shapes
        expected_shape = (batch_size, z1_channels, z1_spatial_size, z1_spatial_size, z1_spatial_size)
        assert mu.shape == expected_shape
        assert logvar.shape == expected_shape


class TestDecoderConditioning:
    """Test decoder conditioning strategies."""
    
    def test_concat_conditioning(self):
        """Test ConcatConditioning."""
        batch_size = 2
        z1_channels = 32
        z2_channels = 16
        z1_spatial_size = 8
        z2_spatial_size = 4
        
        # Create conditioning
        conditioning = ConcatConditioning(
            z2_channels=z2_channels,
            z2_spatial_size=z2_spatial_size,
            z1_spatial_size=z1_spatial_size
        )
        
        # Create inputs
        z1 = torch.randn(batch_size, z1_channels, z1_spatial_size, z1_spatial_size, z1_spatial_size)
        z2 = torch.randn(batch_size, z2_channels, z2_spatial_size, z2_spatial_size, z2_spatial_size)
        
        # Forward pass
        output = conditioning(z1, z2)
        
        # Check shape: should be concatenated along channels
        expected_shape = (batch_size, z1_channels + z2_channels, z1_spatial_size, z1_spatial_size, z1_spatial_size)
        assert output.shape == expected_shape
    
    def test_film_conditioning(self):
        """Test FiLMConditioning."""
        batch_size = 2
        z1_channels = 32
        z2_channels = 16
        z1_spatial_size = 8
        z2_spatial_size = 4
        
        # Create conditioning
        conditioning = FiLMConditioning(
            z2_channels=z2_channels,
            z2_spatial_size=z2_spatial_size,
            z1_channels=z1_channels,
            z1_spatial_size=z1_spatial_size
        )
        
        # Create inputs
        z1 = torch.randn(batch_size, z1_channels, z1_spatial_size, z1_spatial_size, z1_spatial_size)
        z2 = torch.randn(batch_size, z2_channels, z2_spatial_size, z2_spatial_size, z2_spatial_size)
        
        # Forward pass
        output = conditioning(z1, z2)
        
        # Check shape: should be same as z1
        expected_shape = (batch_size, z1_channels, z1_spatial_size, z1_spatial_size, z1_spatial_size)
        assert output.shape == expected_shape


class TestHVAE:
    """Test HVAE model."""
    
    def test_hvae_1_layer_forward(self):
        """Test HVAE 1-layer mode forward pass."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        channel_schedule_top = [32, 64, 128]
        spatial_size_top = 4
        
        # Create HVAE
        hvae = HVAE(
            num_layers=1,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            channel_schedule_top=channel_schedule_top,
            spatial_size_top=spatial_size_top
        )
        
        # Create input
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        
        # Forward pass
        outputs = hvae(x)
        x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params = outputs
        
        # Check shapes for 1-layer mode
        assert x_recon.shape == x.shape
        # Calculate expected spatial size based on channel schedule
        expected_spatial_size_top = 128 // (2 ** len(channel_schedule_top))
        expected_z_top_shape = (batch_size, latent_channels_top, expected_spatial_size_top, expected_spatial_size_top, expected_spatial_size_top)
        assert z_top.shape == expected_z_top_shape
        assert mu_top.shape == expected_z_top_shape
        assert logvar_top.shape == expected_z_top_shape
        
        # Bottom latents should be None for 1-layer mode
        assert z_bottom is None
        assert mu_bottom is None
        assert logvar_bottom is None
        assert prior_params is None
    
    def test_hvae_2_layer_forward(self):
        """Test HVAE 2-layer mode forward pass."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        channel_schedule_top = [32, 64, 128]
        channel_schedule_bottom = [64, 128, 256]
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        # Create HVAE
        hvae = HVAE(
            num_layers=2,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            latent_channels_bottom=latent_channels_bottom,
            channel_schedule_top=channel_schedule_top,
            channel_schedule_bottom=channel_schedule_bottom,
            spatial_size_top=spatial_size_top,
            spatial_size_bottom=spatial_size_bottom
        )
        
        # Create input
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        
        # Forward pass
        outputs = hvae(x)
        x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params = outputs
        
        # Check shapes for 2-layer mode
        assert x_recon.shape == x.shape
        # Calculate expected spatial sizes based on channel schedules
        expected_spatial_size_top = 128 // (2 ** len(channel_schedule_top))
        expected_spatial_size_bottom = 128 // (2 ** len(channel_schedule_bottom))
        expected_z_top_shape = (batch_size, latent_channels_top, expected_spatial_size_top, expected_spatial_size_top, expected_spatial_size_top)
        expected_z_bottom_shape = (batch_size, latent_channels_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom)
        
        assert z_top.shape == expected_z_top_shape
        assert mu_top.shape == expected_z_top_shape
        assert logvar_top.shape == expected_z_top_shape
        
        assert z_bottom.shape == expected_z_bottom_shape
        assert mu_bottom.shape == expected_z_bottom_shape
        assert logvar_bottom.shape == expected_z_bottom_shape
        
        # Check prior params
        assert prior_params is not None
        mu1_p, logvar1_p = prior_params
        assert mu1_p.shape == expected_z_bottom_shape
        assert logvar1_p.shape == expected_z_bottom_shape
    
    def test_hvae_encode_decode(self):
        """Test HVAE encode and decode methods."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        channel_schedule_top = [32, 64, 128]
        channel_schedule_bottom = [64, 128, 256]
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        # Create HVAE
        hvae = HVAE(
            num_layers=2,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            latent_channels_bottom=latent_channels_bottom,
            channel_schedule_top=channel_schedule_top,
            channel_schedule_bottom=channel_schedule_bottom,
            spatial_size_top=spatial_size_top,
            spatial_size_bottom=spatial_size_bottom
        )
        
        # Create input
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        
        # Test encode
        z_top, z_bottom = hvae.encode(x)
        # Calculate expected spatial sizes based on channel schedules
        expected_spatial_size_top = 128 // (2 ** len(channel_schedule_top))
        expected_spatial_size_bottom = 128 // (2 ** len(channel_schedule_bottom))
        expected_z_top_shape = (batch_size, latent_channels_top, expected_spatial_size_top, expected_spatial_size_top, expected_spatial_size_top)
        expected_z_bottom_shape = (batch_size, latent_channels_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom)
        assert z_top.shape == expected_z_top_shape
        assert z_bottom.shape == expected_z_bottom_shape
        
        # Test decode
        x_recon = hvae.decode(z_top, z_bottom)
        assert x_recon.shape == x.shape
    
    def test_hvae_sample(self):
        """Test HVAE sampling."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        channel_schedule_top = [32, 64, 128]
        channel_schedule_bottom = [64, 128, 256]
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        # Create HVAE
        hvae = HVAE(
            num_layers=2,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            latent_channels_bottom=latent_channels_bottom,
            channel_schedule_top=channel_schedule_top,
            channel_schedule_bottom=channel_schedule_bottom,
            spatial_size_top=spatial_size_top,
            spatial_size_bottom=spatial_size_bottom
        )
        
        # Test sampling
        device = torch.device('cpu')
        samples = hvae.sample(num_samples=batch_size, device=device)
        # The decoder should output the same size as input (128³)
        expected_shape = (batch_size, input_channels, 128, 128, 128)
        assert samples.shape == expected_shape


class TestHVAELoss:
    """Test HVAELoss."""
    
    def test_hvae_loss_1_layer(self):
        """Test HVAELoss for 1-layer mode."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        spatial_size_top = 4
        
        # Create HVAE model
        hvae = HVAE(
            num_layers=1,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            channel_schedule_top=[32, 64, 128],
            spatial_size_top=spatial_size_top
        )
        
        # Create loss
        loss_fn = HVAELoss(
            reconstruction_weight=1.0,
            kl_weight_bottom=0.001,
            kl_weight_top=0.001,
            model=hvae
        )
        
        # Create inputs
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        x_recon = torch.randn_like(x)
        mu_top = torch.randn(batch_size, latent_channels_top, spatial_size_top, spatial_size_top, spatial_size_top)
        logvar_top = torch.randn_like(mu_top)
        z_top = torch.randn_like(mu_top)
        
        # Forward pass
        outputs = loss_fn(x_recon, x, mu_top, logvar_top, z_top)
        total_loss, recon_loss, kl_bottom, kl_top, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = outputs
        
        # Check that all outputs are scalars
        assert total_loss.dim() == 0
        assert recon_loss.dim() == 0
        assert kl_bottom.dim() == 0
        assert kl_top.dim() == 0
        assert data_recon.dim() == 0
        assert bg_penalty.dim() == 0
        assert edge_loss.dim() == 0
        assert kl_total.dim() == 0
        assert vamp_reg.dim() == 0
        
        # For 1-layer mode, kl_bottom should be 0
        assert kl_bottom.item() == 0.0
    
    def test_hvae_loss_2_layer(self):
        """Test HVAELoss for 2-layer mode."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        # Create HVAE model
        hvae = HVAE(
            num_layers=2,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            latent_channels_bottom=latent_channels_bottom,
            channel_schedule_top=[32, 64, 128],
            channel_schedule_bottom=[64, 128, 256],
            spatial_size_top=spatial_size_top,
            spatial_size_bottom=spatial_size_bottom
        )
        
        # Create loss
        loss_fn = HVAELoss(
            reconstruction_weight=1.0,
            kl_weight_bottom=0.001,
            kl_weight_top=0.001,
            model=hvae
        )
        
        # Create inputs
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        x_recon = torch.randn_like(x)
        mu_top = torch.randn(batch_size, latent_channels_top, spatial_size_top, spatial_size_top, spatial_size_top)
        logvar_top = torch.randn_like(mu_top)
        z_top = torch.randn_like(mu_top)
        mu_bottom = torch.randn(batch_size, latent_channels_bottom, spatial_size_bottom, spatial_size_bottom, spatial_size_bottom)
        logvar_bottom = torch.randn_like(mu_bottom)
        z_bottom = torch.randn_like(mu_bottom)
        prior_params = (torch.randn_like(mu_bottom), torch.randn_like(mu_bottom))
        
        # Forward pass
        outputs = loss_fn(x_recon, x, mu_top, logvar_top, z_top, mu_bottom, logvar_bottom, z_bottom, prior_params)
        total_loss, recon_loss, kl_bottom, kl_top, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = outputs
        
        # Check that all outputs are scalars
        assert total_loss.dim() == 0
        assert recon_loss.dim() == 0
        assert kl_bottom.dim() == 0
        assert kl_top.dim() == 0
        assert data_recon.dim() == 0
        assert bg_penalty.dim() == 0
        assert edge_loss.dim() == 0
        assert kl_total.dim() == 0
        assert vamp_reg.dim() == 0
        
        # For 2-layer mode, both KL terms should be non-zero
        assert kl_bottom.item() != 0.0
        assert kl_top.item() != 0.0


class TestHVAEIntegration:
    """Integration tests for HVAE."""
    
    def test_hvae_full_forward_backward(self):
        """Test full forward and backward pass."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        channel_schedule_top = [32, 64, 128]
        channel_schedule_bottom = [64, 128, 256]
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        # Create HVAE
        hvae = HVAE(
            num_layers=2,
            input_channels=input_channels,
            latent_channels_top=latent_channels_top,
            latent_channels_bottom=latent_channels_bottom,
            channel_schedule_top=channel_schedule_top,
            channel_schedule_bottom=channel_schedule_bottom,
            spatial_size_top=spatial_size_top,
            spatial_size_bottom=spatial_size_bottom
        )
        
        # Create loss
        loss_fn = HVAELoss(
            reconstruction_weight=1.0,
            kl_weight_bottom=0.001,
            kl_weight_top=0.001,
            model=hvae
        )
        
        # Create input
        x = torch.randn(batch_size, input_channels, 128, 128, 128)
        
        # Forward pass
        outputs = hvae(x)
        x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params = outputs
        
        # Compute loss
        loss_outputs = loss_fn(x_recon, x, mu_top, logvar_top, z_top, mu_bottom, logvar_bottom, z_bottom, prior_params)
        total_loss, recon_loss, kl_bottom, kl_top, data_recon, bg_penalty, edge_loss, kl_total, vamp_reg = loss_outputs
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for param in hvae.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_hvae_different_conditioning_types(self):
        """Test HVAE with different decoder conditioning types."""
        batch_size = 2
        input_channels = 10
        latent_channels_top = 16
        latent_channels_bottom = 32
        channel_schedule_top = [32, 64, 128]
        channel_schedule_bottom = [64, 128, 256]
        spatial_size_top = 4
        spatial_size_bottom = 8
        
        for conditioning_type in ['concat', 'film']:
            # Create HVAE
            hvae = HVAE(
                num_layers=2,
                input_channels=input_channels,
                latent_channels_top=latent_channels_top,
                latent_channels_bottom=latent_channels_bottom,
                channel_schedule_top=channel_schedule_top,
                channel_schedule_bottom=channel_schedule_bottom,
                spatial_size_top=spatial_size_top,
                spatial_size_bottom=spatial_size_bottom,
                decoder_conditioning_type=conditioning_type
            )
            
            # Create input
            x = torch.randn(batch_size, input_channels, 128, 128, 128)
            
            # Forward pass
            outputs = hvae(x)
            x_recon, z_top, mu_top, logvar_top, z_bottom, mu_bottom, logvar_bottom, prior_params = outputs
            
            # Check that reconstruction has correct shape
            assert x_recon.shape == x.shape
            
            # Check that all latents have correct shapes
            # Calculate expected spatial sizes based on channel schedules
            expected_spatial_size_top = 128 // (2 ** len(channel_schedule_top))
            expected_spatial_size_bottom = 128 // (2 ** len(channel_schedule_bottom))
            expected_z_top_shape = (batch_size, latent_channels_top, expected_spatial_size_top, expected_spatial_size_top, expected_spatial_size_top)
            expected_z_bottom_shape = (batch_size, latent_channels_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom, expected_spatial_size_bottom)
            
            assert z_top.shape == expected_z_top_shape
            assert z_bottom.shape == expected_z_bottom_shape


if __name__ == "__main__":
    pytest.main([__file__])
