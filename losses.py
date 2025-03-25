import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
import torch.autograd.functional as AGF

class AdaptiveRegularizationLoss3_1mod(nn.Module):
    def __init__(self, lambda_base, mu_base, lambda_folding=10.0, 
                 beta_adaptive=1.0, beta_norm=1.0, threshold=0.1,
                 center=0.05, scale=0.01):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        
        Args:
            lambda_base (float): Base Lame parameter for strain energy.
            mu_base (float): Base Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
            beta_norm (float): Scale for deformation magnitude-based adjustments.
            threshold (float): Threshold for gradient norm adjustments for lambda_strain.
            
            center (float): The center of the sigmoid function for mu_shear adaptation.
            scale (float): Controls the steepness of the sigmoid function for mu_shear.
        """
        super(AdaptiveRegularizationLoss3_1mod, self).__init__()
        self.lambda_base = lambda_base
        self.mu_base = mu_base
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive
        self.beta_norm = beta_norm
        self.threshold = threshold

        # New parameters for mu_shear's sigmoid adaptation
        self.center = center
        self.scale = scale

    def forward(self, deformation_field, volume):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Compute the norm of the displacement field gradients
        gradient_norm = torch.sqrt(
            dfdx[0] ** 2 + dfdy[0] ** 2 + dfdz[0] ** 2 +
            dfdx[1] ** 2 + dfdy[1] ** 2 + dfdz[1] ** 2 +
            dfdx[2] ** 2 + dfdy[2] ** 2 + dfdz[2] ** 2
        )

        # Dynamic adjustment of lambda_strain based on gradient norm
        lambda_strain = self.lambda_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))
        
        # Using a sigmoid function for mu_shear:
        # mu_shear = mu_base * [1 + beta_norm * sigmoid(-(gradient_norm - center)/scale)]
        mu_shear = self.mu_base * (1 + self.beta_norm * torch.sigmoid(- (gradient_norm - self.center) / self.scale))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density with dynamic weighting
        strain_E = 0.5 * lambda_strain * (trace_E ** 2) 
        shear_E = mu_shear * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        
        elastic_energy_density = strain_E + shear_E

        # Adaptive weighting based on gradient magnitude
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_norm)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss, lambda_strain, mu_shear, adaptive_weight

class CombinedLoss_gabor(nn.Module):
    """
    A loss function that computes NCC between:
    - Original volumes
    - Gabor filter (edge detection) in 3D
    - Wavelet transform (shape detection) in 3D
    """

    def __init__(self, win=None):
        super(CombinedLoss_gabor, self).__init__()
        self.ncc = NCC_vxm(win)

    def gabor_filter(self, volume, theta=(0, 0, 0)):
        """
        Apply a 3D Gabor filter for edge detection in all orientations.
        """
        sigma = 1.0  # Standard deviation of the Gaussian envelope
        lambda_ = 10.0  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0  # Phase offset

        # Create a 3D Gabor kernel
        size = 9  # Gabor kernel size
        x, y, z = torch.meshgrid(
            torch.arange(-size // 2 + 1, size // 2 + 1),
            torch.arange(-size // 2 + 1, size // 2 + 1),
            torch.arange(-size // 2 + 1, size // 2 + 1),
        )
        x, y, z = x.float(), y.float(), z.float()

        # Convert theta to tensor to avoid type errors
        theta_x, theta_y, theta_z = torch.tensor(theta)

        # Gabor function components in 3D
        x_theta = x * torch.cos(theta_x) + y * torch.sin(theta_y) + z * torch.sin(theta_z)
        gb = torch.exp(-0.5 * (x_theta ** 2 / sigma ** 2)) * torch.cos(2 * math.pi * x_theta / lambda_ + psi)

        gb = gb.unsqueeze(0).unsqueeze(0).to(volume.device)  # Convert to 5D tensor for conv3d

        # Apply Gabor filter to the entire 3D volume
        filtered_volume = F.conv3d(volume, gb, padding=size // 2)

        return filtered_volume

    def wavelet_transform(self, volume):
        """
        3D Haar wavelet transform using a 3D Haar wavelet.
        """
        # Haar wavelet (approximate)
        haar_kernel = torch.tensor(
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(volume.device) / 2.0

        # Apply the Haar wavelet filter in 3D
        filtered_volume = F.conv3d(volume, haar_kernel, padding=1, stride=2)
        # Resize the output back to original size using interpolation
        filtered_volume = F.interpolate(filtered_volume, size=volume.shape[2:], mode='trilinear', align_corners=True)

        return filtered_volume

    def forward(self, y_true, y_pred):
        # Original volume NCC
        loss_ncc_original = self.ncc(y_true, y_pred)

        # Gabor filter NCC (edge detection)
        y_true_gabor = self.gabor_filter(y_true)
        y_pred_gabor = self.gabor_filter(y_pred)
        loss_ncc_gabor = self.ncc(y_true_gabor, y_pred_gabor)

        # Wavelet transform NCC (shape detection)
        y_true_wavelet = self.wavelet_transform(y_true)
        y_pred_wavelet = self.wavelet_transform(y_pred)
        loss_ncc_wavelet = self.ncc(y_true_wavelet, y_pred_wavelet)

        # Combine all three losses
        total_loss = 0.8*loss_ncc_original + 0.1*loss_ncc_gabor + 0.1*loss_ncc_wavelet

        return total_loss

import scipy.ndimage as ndi  # For 3D filters

class Filtered_NCC_vxm(torch.nn.Module):
    """
    NCC loss with volumes passed through multiple 3D filters before computing NCC.
    Includes 3D Gaussian, Sobel, LoG, Median filters.
    """

    def __init__(self, win=None, gauss_sigma=1, log_sigma=1, median_size=3):
        super(Filtered_NCC_vxm, self).__init__()
        self.win = win
        self.gauss_sigma = gauss_sigma
        self.log_sigma = log_sigma
        self.median_size = median_size

    def forward(self, y_true, y_pred):
        # Apply filters to both true and predicted volumes
        filtered_y_true = self.apply_filters(y_true)
        filtered_y_pred = self.apply_filters(y_pred)
        
        # Compute NCC for each filter and average
        ncc_loss_total = 0
        for f_y_true, f_y_pred in zip(filtered_y_true, filtered_y_pred):
            ncc_loss_total += self.ncc_loss(f_y_true, f_y_pred)
        
        return ncc_loss_total / len(filtered_y_true)

    def apply_filters(self, volume):
        """
        Apply a set of 3D filters (Gaussian, Sobel, LoG, Median) to the volume.
        """
        gauss_filtered = self.apply_gaussian_filter(volume)
        sobel_filtered = self.apply_sobel_filter(volume)
        log_filtered = self.apply_log_filter(volume)
        median_filtered = self.apply_median_filter(volume)

        return [volume, gauss_filtered, sobel_filtered, log_filtered, median_filtered]

    def apply_gaussian_filter(self, volume):
        """
        Apply 3D Gaussian filter to smooth the volume.
        """
        gauss_filtered = ndi.gaussian_filter(volume.cpu().numpy(), sigma=self.gauss_sigma)
        return torch.tensor(gauss_filtered).to(volume.device)

    def apply_sobel_filter(self, volume):
        """
        Apply 3D Sobel filter for edge detection in 3D.
        """
        sobel_filtered = np.zeros_like(volume.cpu().numpy())
        for axis in range(3):  # Apply Sobel in each spatial axis (x, y, z)
            sobel_filtered += ndi.sobel(volume.cpu().numpy(), axis=axis)
        return torch.tensor(sobel_filtered).to(volume.device)

    def apply_log_filter(self, volume):
        """
        Apply 3D Laplacian of Gaussian (LoG) filter to detect edges while smoothing.
        """
        log_filtered = ndi.gaussian_laplace(volume.cpu().numpy(), sigma=self.log_sigma)
        return torch.tensor(log_filtered).to(volume.device)

    def apply_median_filter(self, volume):
        """
        Apply 3D median filter to reduce noise while preserving edges.
        """
        median_filtered = ndi.median_filter(volume.cpu().numpy(), size=self.median_size)
        return torch.tensor(median_filtered).to(volume.device)

    def ncc_loss(self, Ii, Ji):
        """
        Compute NCC between two volumes Ii and Ji.
        """
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win

        sum_filt = torch.ones([1, 1, *win]).to(Ii.device)
        pad_no = math.floor(win[0] / 2)
        stride = (1,) * ndims
        padding = (pad_no,) * ndims

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class ElasticEnergyLoss(nn.Module):
    def __init__(self, lambda_lame, mu_lame):
        super(ElasticEnergyLoss, self).__init__()
        self.lambda_lame = lambda_lame
        self.mu_lame = mu_lame

    def forward(self, deformation_field, y_true=None):
        # Compute spatial gradients of the deformation field (displacement gradients)
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))  # Gradient of dx along X, Y, Z
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))  # Gradient of dy along X, Y, Z
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))  # Gradient of dz along X, Y, Z

        # Now access the correct components for the strain tensor
        E_xx = dfdx[0]  # Strain component E_xx (gradient along X)
        E_yy = dfdy[1]  # Strain component E_yy (gradient along Y)
        E_zz = dfdz[2]  # Strain component E_zz (gradient along Z)

        # Off-diagonal strain components
        E_xy = 0.5 * (dfdx[1] + dfdy[0])  # Strain component E_xy
        E_xz = 0.5 * (dfdx[2] + dfdz[0])  # Strain component E_xz
        E_yz = 0.5 * (dfdy[2] + dfdz[1])  # Strain component E_yz

        # Trace of the strain tensor (sum of diagonal components)
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density based on isotropic linear elasticity
        elastic_energy_density = (
            0.5 * self.lambda_lame * (trace_E ** 2) +
            self.mu_lame * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )

        # Integrate the elastic energy density over the volume
        return torch.mean(elastic_energy_density)

class JacobianRegularizationLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        """
        Initializes the Jacobian Regularization Loss function.

        Args:
            epsilon (float): Small value to avoid division by zero in the determinant.
        """
        super(JacobianRegularizationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, displacement_field, _):
        """
        Calculates the Jacobian Regularization Loss.

        Args:
            displacement_field (torch.Tensor): Tensor of shape (1, 3, 160, 192, 224),
                                               representing the displacement field.
        
        Returns:
            torch.Tensor: Scalar loss value representing the regularization term.
        """
        # Ensure the displacement field is of the expected shape
        assert displacement_field.shape == (1, 3, 160, 192, 224), "Expected input shape (1, 3, 160, 192, 224)"

        # Calculate spatial gradients and match dimensions
        dx = displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :]
        dy = displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :]
        dz = displacement_field[:, :, :, :, 1:] - displacement_field[:, :, :, :, :-1]

        # Crop gradients to the minimum shape (1, 3, 159, 191, 223) to ensure size compatibility
        min_shape = (dx.shape[0], dx.shape[1], min(dx.shape[2], dy.shape[2], dz.shape[2]),
                     min(dx.shape[3], dy.shape[3], dz.shape[3]), min(dx.shape[4], dy.shape[4], dz.shape[4]))

        dx = dx[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]
        dy = dy[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]
        dz = dz[:, :, :min_shape[2], :min_shape[3], :min_shape[4]]

        # Compute the Jacobian determinant
        jacobian_det = (
            dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1]) -
            dx[:, 1] * (dy[:, 0] * dz[:, 2] - dy[:, 2] * dz[:, 0]) +
            dx[:, 2] * (dy[:, 0] * dz[:, 1] - dy[:, 1] * dz[:, 0])
        )

        # Regularization loss is the variance of the Jacobian determinant
        jacobian_det = jacobian_det + self.epsilon  # To avoid zero determinant issues
        jacobian_loss = torch.var(jacobian_det)

        return jacobian_loss

class MultiScaleAdaptiveElasticityLossWithLame(nn.Module):
    def __init__(self, num_scales, lambda_0, mu_0, kappa_lambda, kappa_mu, base_weight, gradient_scaling):
        """
        Initializes the multi-scale adaptive elasticity loss with spatially varying Lame parameters.

        Args:
            num_scales (int): Number of spatial scales.
            lambda_0 (float): Base value for the first Lame parameter (volumetric resistance).
            mu_0 (float): Base value for the second Lame parameter (shear resistance).
            kappa_lambda (float): Scaling factor for adaptive lambda.
            kappa_mu (float): Scaling factor for adaptive mu.
            base_weight (float): Base weight for the adaptive elasticity regularizer.
            gradient_scaling (float): Scaling factor for the gradient-based adaptivity.
        """
        super(MultiScaleAdaptiveElasticityLossWithLame, self).__init__()
        self.num_scales = num_scales
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.kappa_lambda = kappa_lambda
        self.kappa_mu = kappa_mu
        self.base_weight = base_weight
        self.gradient_scaling = gradient_scaling

    def compute_lame_parameters(self, image_gradient):
        """
        Computes adaptive Lame parameters based on image gradients.

        Args:
            image_gradient (torch.Tensor): Gradient magnitude of the image.

        Returns:
            torch.Tensor, torch.Tensor: Adaptive lambda and mu tensors.
        """
        adaptive_lambda = self.lambda_0 + self.kappa_lambda * image_gradient
        adaptive_mu = self.mu_0 + self.kappa_mu * image_gradient
        return adaptive_lambda, adaptive_mu

    def forward(self, deformation_field, image):
        """
        Computes the multi-scale adaptive elasticity loss with adaptive Lame parameters.

        Args:
            deformation_field (torch.Tensor): Deformation field of shape (batch_size, 3, X, Y, Z).
            image (torch.Tensor): Input image of shape (batch_size, 1, X, Y, Z).

        Returns:
            torch.Tensor: Multi-scale adaptive elasticity loss.
        """
        total_loss = 0.0
        scales = [2 ** i for i in range(self.num_scales)]  # Define scales

        for scale in scales:
            # Downsample deformation field and image
            deform_scaled = F.interpolate(
                deformation_field, scale_factor=1/scale, mode='trilinear',
                align_corners=True, recompute_scale_factor=True
            )
            image_scaled = F.interpolate(
                image, scale_factor=1/scale, mode='trilinear',
                align_corners=True, recompute_scale_factor=True
            )

            # Deformation field components
            u = deform_scaled[:, 0, ...]
            v = deform_scaled[:, 1, ...]
            w = deform_scaled[:, 2, ...]

            # Compute partial derivatives of deformation field components
            du_dx = torch.gradient(u, dim=-3)[0]
            du_dy = torch.gradient(u, dim=-2)[0]
            du_dz = torch.gradient(u, dim=-1)[0]

            dv_dx = torch.gradient(v, dim=-3)[0]
            dv_dy = torch.gradient(v, dim=-2)[0]
            dv_dz = torch.gradient(v, dim=-1)[0]

            dw_dx = torch.gradient(w, dim=-3)[0]
            dw_dy = torch.gradient(w, dim=-2)[0]
            dw_dz = torch.gradient(w, dim=-1)[0]

            # Compute strain tensor components
            E_xx = du_dx
            E_yy = dv_dy
            E_zz = dw_dz
            E_xy = 0.5 * (du_dy + dv_dx)
            E_xz = 0.5 * (du_dz + dw_dx)
            E_yz = 0.5 * (dv_dz + dw_dy)
            trace_E = E_xx + E_yy + E_zz

            # Compute image gradient for adaptivity
            image_channel = image_scaled[:, 0, ...]
            gradients = torch.gradient(image_channel, dim=(-3, -2, -1))
            squared_gradients = [g ** 2 for g in gradients]
            sum_squared_gradients = torch.sum(torch.stack(squared_gradients), dim=0)
            image_gradient = torch.sqrt(sum_squared_gradients)

            # Compute adaptive Lame parameters
            adaptive_lambda, adaptive_mu = self.compute_lame_parameters(image_gradient)

            # Elastic energy density
            elastic_energy = (
                0.5 * adaptive_lambda * (trace_E ** 2) +
                adaptive_mu * (
                    E_xx ** 2 + E_yy ** 2 + E_zz ** 2 +
                    2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2)
                )
            )

            # Compute adaptive weight
            adaptive_weight = self.base_weight + self.gradient_scaling * image_gradient

            # Integrate adaptive elastic energy
            weighted_energy = adaptive_weight * elastic_energy
            total_loss += torch.mean(weighted_energy)

        return total_loss    

class LearnableAdaptiveElasticityLoss(nn.Module):
    def __init__(self, num_scales, lambda_init, mu_init, kappa_lambda_init, kappa_mu_init, base_weight, gradient_scaling):
        """
        Initializes the multi-scale adaptive elasticity loss with learnable Lame parameters and volume information.

        Args:
            num_scales (int): Number of spatial scales.
            lambda_init (float): Initial value for the first Lame parameter (volumetric resistance).
            mu_init (float): Initial value for the second Lame parameter (shear resistance).
            kappa_lambda_init (float): Initial value for adaptive lambda scaling factor.
            kappa_mu_init (float): Initial value for adaptive mu scaling factor.
            base_weight (float): Base weight for the adaptive elasticity regularizer.
            gradient_scaling (float): Scaling factor for the deformation gradient-based adaptivity.
        """
        super(LearnableAdaptiveElasticityLoss, self).__init__()
        
        # Learnable Lame parameters
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.mu_param = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        
        # Learnable scaling factors for adaptivity
        self.kappa_lambda = nn.Parameter(torch.tensor(kappa_lambda_init, dtype=torch.float32))
        self.kappa_mu = nn.Parameter(torch.tensor(kappa_mu_init, dtype=torch.float32))
        
        # Fixed parameters
        self.num_scales = num_scales
        self.base_weight = base_weight
        self.gradient_scaling = gradient_scaling

    def compute_lame_parameters(self, deformation_gradient, volume):
        """
        Computes adaptive Lame parameters based on deformation gradients and volume.

        Args:
            deformation_gradient (torch.Tensor): Gradient magnitude of the deformation field.
            volume (torch.Tensor): Volume mask or class map of shape (batch_size, 1, X, Y, Z).

        Returns:
            torch.Tensor, torch.Tensor: Adaptive lambda and mu tensors.
        """
        # Adjust Lame parameters using the volume information (e.g., scaling based on tissue type)
        adaptive_lambda = (self.lambda_param + self.kappa_lambda * deformation_gradient) * volume
        adaptive_mu = (self.mu_param + self.kappa_mu * deformation_gradient) * volume
        return adaptive_lambda, adaptive_mu

    def forward(self, deformation_field, volume):
        """
        Computes the multi-scale adaptive elasticity loss with learnable Lame parameters and volume information.

        Args:
            deformation_field (torch.Tensor): Deformation field of shape (batch_size, 3, X, Y, Z).
            volume (torch.Tensor): Volume mask or class map of shape (batch_size, 1, X, Y, Z).

        Returns:
            torch.Tensor: Multi-scale adaptive elasticity loss.
        """
        total_loss = 0.0
        scales = [2 ** i for i in range(self.num_scales)]  # Define scales

        for scale in scales:
            # Downsample deformation field and volume
            deform_scaled = F.interpolate(deformation_field, scale_factor=1/scale, mode='trilinear', align_corners=True)
            volume_scaled = F.interpolate(volume, scale_factor=1/scale, mode='trilinear', align_corners=True)

            # Compute gradients of the deformation field
            # Deformation field components
            u = deform_scaled[:, 0, ...]
            v = deform_scaled[:, 1, ...]
            w = deform_scaled[:, 2, ...]
    
            # Compute partial derivatives of deformation field components
            du_dx = torch.gradient(u, dim=-3)[0]
            du_dy = torch.gradient(u, dim=-2)[0]
            du_dz = torch.gradient(u, dim=-1)[0]
    
            dv_dx = torch.gradient(v, dim=-3)[0]
            dv_dy = torch.gradient(v, dim=-2)[0]
            dv_dz = torch.gradient(v, dim=-1)[0]
    
            dw_dx = torch.gradient(w, dim=-3)[0]
            dw_dy = torch.gradient(w, dim=-2)[0]
            dw_dz = torch.gradient(w, dim=-1)[0]
    
            # Compute strain tensor components
            E_xx = du_dx
            E_yy = dv_dy
            E_zz = dw_dz
            E_xy = 0.5 * (du_dy + dv_dx)
            E_xz = 0.5 * (du_dz + dw_dx)
            E_yz = 0.5 * (dv_dz + dw_dy)
            trace_E = E_xx + E_yy + E_zz
            # Compute deformation gradient magnitude for adaptivity
            deformation_gradient = torch.sqrt(E_xx**2 + E_yy**2 + E_zz**2 + 2 * (E_xy**2 + E_xz**2 + E_yz**2))

            # Compute adaptive Lame parameters using volume
            adaptive_lambda, adaptive_mu = self.compute_lame_parameters(deformation_gradient, volume_scaled)

            # Elastic energy density
            elastic_energy = (
                0.5 * adaptive_lambda * (trace_E ** 2) +
                adaptive_mu * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
            )

            # Compute adaptive weight based on deformation gradient magnitude
            adaptive_weight = self.base_weight + self.gradient_scaling * deformation_gradient

            # Integrate adaptive elastic energy
            weighted_energy = adaptive_weight * elastic_energy
            total_loss += torch.mean(weighted_energy)

        return total_loss

class EnhancedMultiScaleAdaptiveElasticityLossWithLame(nn.Module):
    def __init__(self, num_scales, lambda_0, mu_0, kappa_lambda, kappa_mu, 
                 base_weight, gradient_scaling, 
                 clamp_min=0.1, clamp_max=10.0, 
                 scale_weights=None, jacobian_penalty_weight=0.1):
        """
        Enhanced version of the multi-scale adaptive elasticity loss with improvements.

        Args:
            num_scales (int): Number of spatial scales.
            lambda_0 (float): Base value for the first Lame parameter (volumetric resistance).
            mu_0 (float): Base value for the second Lame parameter (shear resistance).
            kappa_lambda (float): Scaling factor for adaptive lambda.
            kappa_mu (float): Scaling factor for adaptive mu.
            base_weight (float): Base weight for the adaptive elasticity regularizer.
            gradient_scaling (float): Scaling factor for the gradient-based adaptivity.
            clamp_min (float): Minimum value for adaptive Lame parameters.
            clamp_max (float): Maximum value for adaptive Lame parameters.
            scale_weights (list): Weights for multi-scale contributions.
            jacobian_penalty_weight (float): Weight for the Jacobian determinant penalty.
        """
        super(EnhancedMultiScaleAdaptiveElasticityLossWithLame, self).__init__()
        self.num_scales = num_scales
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.kappa_lambda = kappa_lambda
        self.kappa_mu = kappa_mu
        self.base_weight = base_weight
        self.gradient_scaling = gradient_scaling
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale_weights = scale_weights
        self.jacobian_penalty_weight = jacobian_penalty_weight

    def compute_lame_parameters(self, image_gradient):
        """
        Computes adaptive Lame parameters with clamping to stabilize the values.

        Args:
            image_gradient (torch.Tensor): Gradient magnitude of the image.

        Returns:
            torch.Tensor, torch.Tensor: Clamped adaptive lambda and mu tensors.
        """
        adaptive_lambda = torch.clamp(
            self.lambda_0 + self.kappa_lambda * image_gradient, 
            self.clamp_min, self.clamp_max
        )
        adaptive_mu = torch.clamp(
            self.mu_0 + self.kappa_mu * image_gradient, 
            self.clamp_min, self.clamp_max
        )
        return adaptive_lambda, adaptive_mu

    def compute_jacobian_penalty(self, deformation_field):
        """
        Computes the penalty for negative Jacobian determinants.

        Args:
            deformation_field (torch.Tensor): Deformation field.

        Returns:
            torch.Tensor: Jacobian determinant penalty.
        """
        batch_size, _, X, Y, Z = deformation_field.shape
        jacobian_dets = []
        
        # Compute Jacobian determinant for each voxel in the deformation field
        for b in range(batch_size):
            jacobian = AGF.jacobian(
                lambda coords: deformation_field[b, :, coords[0], coords[1], coords[2]],
                torch.tensor([X // 2, Y // 2, Z // 2], requires_grad=True).int()
            )
            jacobian_det = torch.det(jacobian)
            jacobian_dets.append(jacobian_det)

        # Penalize negative Jacobians
        jacobian_penalty = torch.mean(F.relu(-torch.stack(jacobian_dets)))
        return jacobian_penalty

    def forward(self, deformation_field, image):
        """
        Computes the enhanced multi-scale adaptive elasticity loss.

        Args:
            deformation_field (torch.Tensor): Deformation field of shape (batch_size, 3, X, Y, Z).
            image (torch.Tensor): Input image of shape (batch_size, 1, X, Y, Z).

        Returns:
            torch.Tensor: Total enhanced loss.
        """
        total_loss = 0.0
        scales = [2 ** i for i in range(self.num_scales)]  # Define scales

        for i, scale in enumerate(scales):
            # Downsample deformation field and image
            deform_scaled = F.interpolate(
                deformation_field, scale_factor=1/scale, mode='trilinear',
                align_corners=True, recompute_scale_factor=True
            )
            image_scaled = F.interpolate(
                image, scale_factor=1/scale, mode='trilinear',
                align_corners=True, recompute_scale_factor=True
            )

            # Compute image gradient (apply smoothing filter if needed)
            image_channel = image_scaled[:, 0, ...]
            gradients = torch.gradient(image_channel, dim=(-3, -2, -1))
            squared_gradients = [g ** 2 for g in gradients]
            sum_squared_gradients = torch.sum(torch.stack(squared_gradients), dim=0)
            image_gradient = torch.sqrt(sum_squared_gradients)
            image_gradient_smoothed = F.gaussian_blur(image_gradient.unsqueeze(0), kernel_size=(5, 5, 5))

            # Compute adaptive Lame parameters
            adaptive_lambda, adaptive_mu = self.compute_lame_parameters(image_gradient_smoothed)

            # Compute strain tensor components
            du_dx, du_dy, du_dz = torch.gradient(deform_scaled[:, 0, ...], dim=(-3, -2, -1))
            dv_dx, dv_dy, dv_dz = torch.gradient(deform_scaled[:, 1, ...], dim=(-3, -2, -1))
            dw_dx, dw_dy, dw_dz = torch.gradient(deform_scaled[:, 2, ...], dim=(-3, -2, -1))
            E_xx, E_yy, E_zz = du_dx, dv_dy, dw_dz
            E_xy, E_xz, E_yz = 0.5 * (du_dy + dv_dx), 0.5 * (du_dz + dw_dx), 0.5 * (dv_dz + dw_dy)
            trace_E = E_xx + E_yy + E_zz

            # Compute elastic energy density
            elastic_energy = (
                0.5 * adaptive_lambda * (trace_E ** 2) +
                adaptive_mu * (
                    E_xx ** 2 + E_yy ** 2 + E_zz ** 2 +
                    2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2)
                )
            )

            # Adaptive weighting
            adaptive_weight = self.base_weight + self.gradient_scaling * image_gradient_smoothed
            weighted_energy = adaptive_weight * elastic_energy
            total_loss += self.scale_weights[i] * torch.mean(weighted_energy)

        # Add Jacobian determinant penalty
        jacobian_penalty = self.compute_jacobian_penalty(deformation_field)
        total_loss += self.jacobian_penalty_weight * jacobian_penalty

        return total_loss

class MSElasticEnergyLoss(nn.Module):
    def __init__(self, lambda_lame, mu_lame, scales=[1, 2, 4]):
        """
        Elastic Energy Loss with Multi-Scale only.
        Args:
            lambda_lame (float): First Lame parameter (bulk modulus).
            mu_lame (float): Second Lame parameter (shear modulus).
            scales (list): List of scale factors for multi-scale regularization.
        """
        super(MSElasticEnergyLoss, self).__init__()
        self.lambda_lame = lambda_lame
        self.mu_lame = mu_lame
        self.scales = scales

    def compute_elastic_energy(self, deformation_field):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))  # Gradient of dx along X, Y, Z
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))  # Gradient of dy along X, Y, Z
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))  # Gradient of dz along X, Y, Z

        # Strain tensor components
        E_xx = dfdx[0]  # Gradient of dx along X
        E_yy = dfdy[1]  # Gradient of dy along Y
        E_zz = dfdz[2]  # Gradient of dz along Z

        E_xy = 0.5 * (dfdx[1] + dfdy[0])  # Average of dx/dy and dy/dx
        E_xz = 0.5 * (dfdx[2] + dfdz[0])  # Average of dx/dz and dz/dx
        E_yz = 0.5 * (dfdy[2] + dfdz[1])  # Average of dy/dz and dz/dy

        # Trace of the strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density
        elastic_energy_density = (
            0.5 * self.lambda_lame * (trace_E ** 2) +
            self.mu_lame * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )
        return torch.mean(elastic_energy_density)

    def forward(self, deformation_field, y_true=None):
        """
        Compute the multi-scale elastic energy loss.
        Args:
            deformation_field (Tensor): Deformation field (B, 3, D, H, W).
        Returns:
            Tensor: Multi-scale elastic energy loss.
        """
        multi_scale_loss = 0.0
        for scale in self.scales:
            # Downsample deformation field for each scale
            if scale > 1:
                downsampled_field = torch.nn.functional.interpolate(
                    deformation_field, scale_factor=1/scale, mode='trilinear', align_corners=True
                )
            else:
                downsampled_field = deformation_field

            # Compute elastic energy at the current scale
            scale_loss = self.compute_elastic_energy(downsampled_field)
            multi_scale_loss += scale_loss / len(self.scales)  # Average over scales

        return multi_scale_loss

class AdaptiveRegularizationLoss(nn.Module):
    def __init__(self, lambda_lame, mu_lame, lambda_folding=10.0, beta_adaptive=1.0):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        Args:
            lambda_lame (float): Lame parameter for elastic energy.
            mu_lame (float): Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
        """
        super(AdaptiveRegularizationLoss, self).__init__()
        self.lambda_lame = lambda_lame
        self.mu_lame = mu_lame
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive

    def forward(self, deformation_field, y_true=None):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density
        elastic_energy_density = (
            0.5 * self.lambda_lame * (trace_E ** 2) +
            self.mu_lame * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )

        # Adaptive weighting based on gradient magnitude
        gradient_magnitude = torch.sqrt(E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_magnitude)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss

class ElasticEnergyLossWithJacobian(nn.Module):
    def __init__(self, lambda_lame=1.0, mu_lame=0.5, jacobian_weight=0.1, error_bound=0.01):
        super(ElasticEnergyLossWithJacobian, self).__init__()
        self.lambda_lame = lambda_lame          # Lame parameter lambda
        self.mu_lame = mu_lame                  # Lame parameter mu
        self.jacobian_weight = jacobian_weight  # Weight for the Jacobian penalty term
        self.error_bound = error_bound          # Error bound for the robust loss (delta in Huber loss)

    def forward(self, deformation_field, y_true=None):
        # deformation_field shape: [batch_size, 3, H, W, D]
        batch_size = deformation_field.shape[0]

        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))  # Gradients of u_x
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))  # Gradients of u_y
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))  # Gradients of u_z

        # Access the correct components for the strain tensor
        E_xx = dfdx[0]  # du_x/dx
        E_yy = dfdy[1]  # du_y/dy
        E_zz = dfdz[2]  # du_z/dz

        # Off-diagonal strain components (symmetrized)
        E_xy = 0.5 * (dfdx[1] + dfdy[0])  # 0.5 * (du_x/dy + du_y/dx)
        E_xz = 0.5 * (dfdx[2] + dfdz[0])  # 0.5 * (du_x/dz + du_z/dx)
        E_yz = 0.5 * (dfdy[2] + dfdz[1])  # 0.5 * (du_y/dz + du_z/dy)

        # Trace of the strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Robust loss function (Huber loss)
        def robust_loss(x, delta):
            abs_x = torch.abs(x)
            quadratic = torch.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))
            return quadratic

        # Apply robust loss to strain components
        E_xx_loss = robust_loss(E_xx, self.error_bound)
        E_yy_loss = robust_loss(E_yy, self.error_bound)
        E_zz_loss = robust_loss(E_zz, self.error_bound)
        E_xy_loss = robust_loss(E_xy, self.error_bound)
        E_xz_loss = robust_loss(E_xz, self.error_bound)
        E_yz_loss = robust_loss(E_yz, self.error_bound)
        trace_E_loss = robust_loss(trace_E, self.error_bound)

        # Elastic energy density with robust loss
        elastic_energy_density = (
            0.5 * self.lambda_lame * (trace_E_loss) ** 2 +
            self.mu_lame * (
                E_xx_loss ** 2 + E_yy_loss ** 2 + E_zz_loss ** 2 +
                2 * (E_xy_loss ** 2 + E_xz_loss ** 2 + E_yz_loss ** 2)
            )
        )

        # Compute Jacobian determinant at each voxel
        jacobian_det = self.compute_jacobian_determinant(deformation_field)

        # Penalize negative Jacobian determinants (folding)
        jacobian_penalty = torch.relu(-jacobian_det)  # Penalizes only where determinant is negative

        # Compute mean elastic energy density and Jacobian penalty
        mean_elastic_energy = torch.mean(elastic_energy_density)
        mean_jacobian_penalty = torch.mean(jacobian_penalty)

        # Total loss
        total_loss = mean_elastic_energy + self.jacobian_weight * mean_jacobian_penalty

        return total_loss

    @staticmethod
    def compute_jacobian_determinant(deformation_field):
        # deformation_field shape: [batch_size, 3, H, W, D]
        batch_size = deformation_field.shape[0]

        # Compute gradients of each displacement component
        du_dx = torch.gradient(deformation_field[:, 0, ...], dim=-3)[0]
        du_dy = torch.gradient(deformation_field[:, 0, ...], dim=-2)[0]
        du_dz = torch.gradient(deformation_field[:, 0, ...], dim=-1)[0]

        dv_dx = torch.gradient(deformation_field[:, 1, ...], dim=-3)[0]
        dv_dy = torch.gradient(deformation_field[:, 1, ...], dim=-2)[0]
        dv_dz = torch.gradient(deformation_field[:, 1, ...], dim=-1)[0]

        dw_dx = torch.gradient(deformation_field[:, 2, ...], dim=-3)[0]
        dw_dy = torch.gradient(deformation_field[:, 2, ...], dim=-2)[0]
        dw_dz = torch.gradient(deformation_field[:, 2, ...], dim=-1)[0]

        # Compute Jacobian determinant using the formula for 3D transformations
        jacobian_det = (
            (1 + du_dx) * ((1 + dv_dy) * (1 + dw_dz) - dv_dz * dw_dy) -
            du_dy * (dv_dx * (1 + dw_dz) - dv_dz * (1 + dw_dx)) +
            du_dz * (dv_dx * dw_dy - (1 + dv_dy) * (1 + dw_dx))
        )

        return jacobian_det

class AdaptiveRegularizationLoss2(nn.Module):
    def __init__(self, lambda_lame, mu_lame, lambda_folding=10.0, beta_adaptive=1.0):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        Args:
            lambda_lame (float): Lame parameter for elastic energy.
            mu_lame (float): Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
        """
        super(AdaptiveRegularizationLoss2, self).__init__()
        self.lambda_lame = lambda_lame
        self.mu_lame = mu_lame
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive

    def forward(self, deformation_field, y_true=None):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density
        elastic_energy_density = (
            0.5 * self.lambda_lame * (trace_E ** 2) +
            self.mu_lame * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )

        # Adaptive weighting based on gradient magnitude
        gradient_magnitude = torch.sqrt(E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_magnitude)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss

class AdaptiveRegularizationLoss3(nn.Module):
    def __init__(self, lambda_base, mu_base, lambda_folding=10.0, beta_adaptive=1.0, beta_norm=1.0, threshold=0.1):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        Args:
            lambda_base (float): Base Lame parameter for strain energy.
            mu_base (float): Base Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
            beta_norm (float): Scale for deformation magnitude-based adjustments.
            threshold (float): Threshold for gradient norm adjustments.
        """
        super(AdaptiveRegularizationLoss3, self).__init__()
        self.lambda_base = lambda_base
        self.mu_base = mu_base
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive
        self.beta_norm = beta_norm
        self.threshold = threshold

    def forward(self, deformation_field, volume):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Compute the norm of the displacement field gradients
        gradient_norm = torch.sqrt(
            dfdx[0] ** 2 + dfdy[0] ** 2 + dfdz[0] ** 2 +
            dfdx[1] ** 2 + dfdy[1] ** 2 + dfdz[1] ** 2 +
            dfdx[2] ** 2 + dfdy[2] ** 2 + dfdz[2] ** 2
        )

        # Dynamic adjustment of strain and shear weights based on gradient norm
        lambda_strain = self.lambda_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))
        mu_shear = self.mu_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density with dynamic weighting
        elastic_energy_density = (
            0.5 * lambda_strain * (trace_E ** 2) +
            mu_shear * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )

        # Adaptive weighting based on gradient magnitude
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_norm)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss

class AdaptiveRegularizationLoss3(nn.Module):
    def __init__(self, lambda_base, mu_base, lambda_folding=10.0, beta_adaptive=1.0, beta_norm=1.0, threshold=0.1):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        Args:
            lambda_base (float): Base Lame parameter for strain energy.
            mu_base (float): Base Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
            beta_norm (float): Scale for deformation magnitude-based adjustments.
            threshold (float): Threshold for gradient norm adjustments.
        """
        super(AdaptiveRegularizationLoss3, self).__init__()
        self.lambda_base = lambda_base
        self.mu_base = mu_base
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive
        self.beta_norm = beta_norm
        self.threshold = threshold

    def forward(self, deformation_field, volume):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Compute the norm of the displacement field gradients
        gradient_norm = torch.sqrt(
            dfdx[0] ** 2 + dfdy[0] ** 2 + dfdz[0] ** 2 +
            dfdx[1] ** 2 + dfdy[1] ** 2 + dfdz[1] ** 2 +
            dfdx[2] ** 2 + dfdy[2] ** 2 + dfdz[2] ** 2
        )

        # Dynamic adjustment of strain and shear weights based on gradient norm
        lambda_strain = self.lambda_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))
        mu_shear = self.mu_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density with dynamic weighting
        elastic_energy_density = (
            0.5 * lambda_strain * (trace_E ** 2) +
            mu_shear * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        )

        # Adaptive weighting based on gradient magnitude
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_norm)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss
    
class AdaptiveRegularizationLoss3_1(nn.Module):
    def __init__(self, lambda_base, mu_base, lambda_folding=10.0, 
                 beta_adaptive=1.0, beta_norm=1.0, threshold=0.1,
                 center=0.05, scale=0.01):
        """
        Adaptive Regularization with Folding Prevention and Gradient Weighting.
        
        Args:
            lambda_base (float): Base Lame parameter for strain energy.
            mu_base (float): Base Shear modulus for elastic energy.
            lambda_folding (float): Weight for folding penalty.
            beta_adaptive (float): Scale for adaptive regularization weights.
            beta_norm (float): Scale for deformation magnitude-based adjustments.
            threshold (float): Threshold for gradient norm adjustments for lambda_strain.
            
            center (float): The center of the sigmoid function for mu_shear adaptation.
            scale (float): Controls the steepness of the sigmoid function for mu_shear.
        """
        super(AdaptiveRegularizationLoss3_1, self).__init__()
        self.lambda_base = lambda_base
        self.mu_base = mu_base
        self.lambda_folding = lambda_folding
        self.beta_adaptive = beta_adaptive
        self.beta_norm = beta_norm
        self.threshold = threshold

        # New parameters for mu_shear's sigmoid adaptation
        self.center = center
        self.scale = scale

    def forward(self, deformation_field, volume):
        # Compute spatial gradients of the deformation field
        dfdx = torch.gradient(deformation_field[:, 0, ...], dim=(-3, -2, -1))
        dfdy = torch.gradient(deformation_field[:, 1, ...], dim=(-3, -2, -1))
        dfdz = torch.gradient(deformation_field[:, 2, ...], dim=(-3, -2, -1))

        # Compute the norm of the displacement field gradients
        gradient_norm = torch.sqrt(
            dfdx[0] ** 2 + dfdy[0] ** 2 + dfdz[0] ** 2 +
            dfdx[1] ** 2 + dfdy[1] ** 2 + dfdz[1] ** 2 +
            dfdx[2] ** 2 + dfdy[2] ** 2 + dfdz[2] ** 2
        )

        # Dynamic adjustment of lambda_strain based on gradient norm
        lambda_strain = self.lambda_base * (1 + self.beta_norm * torch.exp(-gradient_norm / self.threshold))
        
        # Using a sigmoid function for mu_shear:
        # mu_shear = mu_base * [1 + beta_norm * sigmoid(-(gradient_norm - center)/scale)]
        mu_shear = self.mu_base * (1 + self.beta_norm * torch.sigmoid(- (gradient_norm - self.center) / self.scale))

        # Strain tensor components
        E_xx = dfdx[0]
        E_yy = dfdy[1]
        E_zz = dfdz[2]
        E_xy = 0.5 * (dfdx[1] + dfdy[0])
        E_xz = 0.5 * (dfdx[2] + dfdz[0])
        E_yz = 0.5 * (dfdy[2] + dfdz[1])

        # Trace of strain tensor
        trace_E = E_xx + E_yy + E_zz

        # Elastic energy density with dynamic weighting
        strain_E = 0.5 * lambda_strain * (trace_E ** 2) 
        shear_E = mu_shear * (E_xx ** 2 + E_yy ** 2 + E_zz ** 2 + 2 * (E_xy ** 2 + E_xz ** 2 + E_yz ** 2))
        
        elastic_energy_density = strain_E + shear_E

        # Adaptive weighting based on gradient magnitude
        adaptive_weight = 1 + self.beta_adaptive * torch.exp(-gradient_norm)
        adaptive_regularization = adaptive_weight * elastic_energy_density

        # Folding prevention (negative Jacobian)
        jacobian = (
            dfdx[0] * (dfdy[1] * dfdz[2] - dfdy[2] * dfdz[1]) -
            dfdx[1] * (dfdy[0] * dfdz[2] - dfdy[2] * dfdz[0]) +
            dfdx[2] * (dfdy[0] * dfdz[1] - dfdy[1] * dfdz[0])
        )
        folding_penalty = torch.mean(torch.relu(-jacobian) ** 2)

        # Combine losses
        total_loss = torch.mean(adaptive_regularization) + self.lambda_folding * folding_penalty

        return total_loss, torch.mean(lambda_strain), torch.mean(mu_shear), torch.mean(gradient_norm),torch.mean(trace_E),torch.mean(strain_E),torch.mean(shear_E),torch.mean(self.lambda_folding * folding_penalty)





def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy

class DiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, num_class=36):
        super().__init__()
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        #y_pred = torch.round(y_pred)
        #y_pred = nn.functional.one_hot(torch.round(y_pred).long(), num_classes=7)
        #y_pred = torch.squeeze(y_pred, 1)
        #y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))
        return dsc

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross-correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(Ii.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class CombinedLoss(nn.Module):
    """
    A loss function that computes NCC between:
    - Original volumes
    - Edge-detected volumes (using Laplacian)
    - Shape-detected volumes (using Sobel filters)
    """

    def __init__(self, win=None):
        super(CombinedLoss, self).__init__()
        self.ncc = NCC_vxm(win)

        # Define Laplacian kernel for edge detection (3x3x3 kernel for 3D)
        self.laplacian_kernel = torch.tensor([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                                              [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
                                              [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Define Sobel kernels for shape detection (3x3x3 kernels for 3D)
        self.sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_kernel_y = torch.tensor([[[-1, -2, -1], [-1, -2, -1], [-1, -2, -1]],
                                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                            [[1, 2, 1], [1, 2, 1], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_kernel_z = torch.tensor([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                                            [[-2, -2, -2], [0, 0, 0], [2, 2, 2]],
                                            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def edge_detection(self, volume):
        """
        Apply Laplacian for edge detection.
        """
        laplacian = F.conv3d(volume, self.laplacian_kernel.to(volume.device), padding=1)
        return laplacian

    def shape_detection(self, volume):
        """
        Apply Sobel filters for shape detection (gradient magnitude).
        """
        grad_x = F.conv3d(volume, self.sobel_kernel_x.to(volume.device), padding=1)
        grad_y = F.conv3d(volume, self.sobel_kernel_y.to(volume.device), padding=1)
        grad_z = F.conv3d(volume, self.sobel_kernel_z.to(volume.device), padding=1)

        # Combine the gradients to get the magnitude of the gradient (shape detection)
        sobel = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        return sobel

    def forward(self, y_true, y_pred):
        # Original volume NCC
        loss_ncc_original = self.ncc(y_true, y_pred)

        # Edge detection NCC
        y_true_edges = self.edge_detection(y_true)
        y_pred_edges = self.edge_detection(y_pred)
        loss_ncc_edges = self.ncc(y_true_edges, y_pred_edges)

        # Shape detection NCC
        y_true_shape = self.shape_detection(y_true)
        y_pred_shape = self.shape_detection(y_pred)
        loss_ncc_shape = self.ncc(y_true_shape, y_pred_shape)

        # Combine all three losses (you can weight them as needed)
        total_loss = 0.8*loss_ncc_original + 0.1*loss_ncc_edges + 0.1*loss_ncc_shape

        return total_loss

class KLDivergenceLoss(torch.nn.Module):
    """
    KL Divergence loss between two volumes (y_true and y_pred).
    """

    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # Flatten the input to ensure correct shape for KL divergence
        Ii_flat = Ii.view(Ii.size(0), -1)  # Flatten across the batch dimension
        Ji_flat = Ji.view(Ji.size(0), -1)

        # Add a small epsilon to avoid division by zero in log calculations
        epsilon = 1e-10

        # Normalize inputs to create valid probability distributions
        P = Ii_flat / (Ii_flat.sum(dim=1, keepdim=True) + epsilon)  # Normalize along the flattened dimensions
        Q = Ji_flat / (Ji_flat.sum(dim=1, keepdim=True) + epsilon)

        # Apply log to Q with epsilon to avoid log(0)
        log_Q = torch.log(Q + epsilon)

        # Compute KL Divergence (P || Q)
        kl_div = torch.sum(P * (torch.log(P + epsilon) - log_Q), dim=1)

        # Return the mean KL divergence for the batch
        return torch.mean(kl_div)

