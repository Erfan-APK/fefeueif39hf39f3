import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from scipy.interpolate import griddata
import time
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings('ignore')

# Set device and precision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print(f"Using device: {device}")


class NACA0012:
    """NACA0012 airfoil geometry"""

    def __init__(self, num_points=400):  # Doubled for better surface resolution
        self.chord = 1.0
        self.num_points = num_points

    def surface_points(self):
        """Generate points on airfoil surface"""
        # Use cosine spacing for better leading edge resolution
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x = 0.5 * (1 - np.cos(theta))

        # NACA0012 thickness distribution
        t = 0.12  # Maximum thickness
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1015 * x ** 4)

        # Upper and lower surface
        x_upper = x[:self.num_points // 2]
        y_upper = yt[:self.num_points // 2]
        x_lower = x[:self.num_points // 2]
        y_lower = -yt[:self.num_points // 2]

        # Combine surfaces
        x_surf = np.concatenate([x_upper, x_lower[::-1]])
        y_surf = np.concatenate([y_upper, y_lower[::-1]])

        return x_surf, y_surf

    def is_inside(self, x, y):
        """Check if point is inside airfoil"""
        if x < 0 or x > self.chord:
            return False

        t = 0.12
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        return abs(y) <= yt


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for better representation"""

    def __init__(self, input_dim, num_features, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.scale = scale

        # Random Fourier features
        self.B = nn.Parameter(torch.randn(input_dim, num_features) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ModifiedMLP(nn.Module):
    """Modified MLP with residual connections and Fourier features"""

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=3, num_layers=8,
                 fourier_features=64, activation='tanh'):
        super().__init__()

        # Fourier feature embedding
        self.fourier = FourierFeatureEmbedding(input_dim, fourier_features, scale=5.0)

        # Network architecture
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(fourier_features * 2, hidden_dim))

        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        # Initialize weights using Xavier initialization
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Fourier feature embedding
        x = self.fourier(x)

        # First layer
        out = self.activation(self.layers[0](x))

        # Hidden layers with residual connections
        for i in range(1, len(self.layers) - 1):
            residual = out
            out = self.activation(self.layers[i](out))
            if i % 2 == 0:  # Add residual every 2 layers
                out = out + residual

        # Output layer (no activation)
        out = self.layers[-1](out)

        return out


class AdaptiveWeighting:
    """Adaptive weighting strategies for loss terms"""

    def __init__(self, method='gradnorm'):
        self.method = method
        self.alpha = 0.9  # Exponential moving average factor
        self.weights = None
        self.prev_losses = None
        self.initial_losses = None  # Store initial losses for normalization
        self.iteration = 0

    def compute_weights(self, losses, model=None):
        """Compute adaptive weights based on losses"""
        # Convert losses to float values (CPU) for computation
        losses_dict = {k: v.detach().cpu().item() if torch.is_tensor(v) else float(v)
                       for k, v in losses.items()}

        self.iteration += 1

        if self.method == 'gradnorm':
            # Initialize on first call
            if self.weights is None:
                self.weights = {k: 1.0 for k in losses_dict.keys()}
                self.prev_losses = losses_dict.copy()
                self.initial_losses = losses_dict.copy()
                return self.weights

            # Skip update for very early iterations to let losses stabilize
            if self.iteration < 10:
                self.prev_losses = losses_dict.copy()
                return self.weights

            # Compute loss ratios (current/previous)
            loss_ratios = {}
            for k in losses_dict.keys():
                if self.prev_losses[k] > 1e-8:  # Avoid division by zero
                    # Normalize by initial loss to handle different scales
                    if self.initial_losses[k] > 1e-8:
                        normalized_current = losses_dict[k] / self.initial_losses[k]
                        normalized_prev = self.prev_losses[k] / self.initial_losses[k]
                        loss_ratios[k] = normalized_current / (normalized_prev + 1e-8)
                    else:
                        loss_ratios[k] = 1.0
                else:
                    loss_ratios[k] = 1.0

            # Compute mean ratio for normalization
            mean_ratio = np.mean(list(loss_ratios.values()))

            # Update weights with smooth adjustment
            # Use smaller exponent for more stable updates
            adjustment_strength = 0.05  # Reduced from 0.1 for stability
            for k in self.weights.keys():
                # Relative rate of change
                relative_rate = (loss_ratios[k] / (mean_ratio + 1e-8))

                # Apply bounded adjustment to prevent extreme weights
                adjustment = np.clip(relative_rate ** adjustment_strength, 0.5, 2.0)
                self.weights[k] *= adjustment

                # Bound individual weights to prevent any single loss from dominating
                self.weights[k] = np.clip(self.weights[k], 0.1, 10.0)

            # Normalize weights to maintain consistent total contribution
            total_weight = sum(self.weights.values())
            num_losses = len(self.weights)
            self.weights = {k: (v / total_weight) * num_losses
                            for k, v in self.weights.items()}

            # Update previous losses with exponential moving average for smoothing
            for k in self.prev_losses.keys():
                self.prev_losses[k] = (self.alpha * self.prev_losses[k] +
                                       (1 - self.alpha) * losses_dict[k])

        elif self.method == 'uncertainty':
            # Uncertainty-based weighting with initialization
            if self.weights is None:
                self.weights = {k: 1.0 for k in losses_dict.keys()}
                self.initial_losses = losses_dict.copy()

            # Normalize losses by initial values
            normalized_losses = {}
            for k, v in losses_dict.items():
                if self.initial_losses[k] > 1e-8:
                    normalized_losses[k] = v / self.initial_losses[k]
                else:
                    normalized_losses[k] = v

            # Use inverse of normalized loss as weight
            total = sum(1.0 / (v + 1e-6) for v in normalized_losses.values())
            self.weights = {k: (1.0 / (v + 1e-6)) / total * len(losses_dict)
                            for k, v in normalized_losses.items()}

            # Apply bounds to weights
            for k in self.weights.keys():
                self.weights[k] = np.clip(self.weights[k], 0.1, 10.0)

        return self.weights


class AdaptiveSampling:
    """Adaptive sampling based on residuals"""

    def __init__(self, initial_points=5000, max_points=20000, add_points=500):
        self.initial_points = initial_points
        self.max_points = max_points
        self.add_points = add_points
        self.current_points = None

    def generate_initial_points(self, domain_bounds, airfoil):
        """Generate initial collocation points"""
        x_min, x_max = domain_bounds['x']
        y_min, y_max = domain_bounds['y']

        points = []
        while len(points) < self.initial_points:
            x = np.random.uniform(x_min, x_max, 1000)
            y = np.random.uniform(y_min, y_max, 1000)

            # Filter out points inside airfoil
            valid_mask = np.array([not airfoil.is_inside(xi, yi)
                                   for xi, yi in zip(x, y)])
            valid_points = np.stack([x[valid_mask], y[valid_mask]], axis=1)
            points.append(valid_points)

            if sum(len(p) for p in points) >= self.initial_points:
                break

        points = np.vstack(points)[:self.initial_points]
        self.current_points = points
        return points

    def add_new_points(self, residuals, domain_bounds, airfoil):
        """Add new points in regions with high residuals"""
        if len(self.current_points) >= self.max_points:
            return self.current_points

        # Find regions with high residuals
        residual_vals = residuals.detach().cpu().numpy()
        threshold = np.percentile(np.abs(residual_vals), 90)
        high_residual_mask = np.abs(residual_vals) > threshold
        high_residual_points = self.current_points[high_residual_mask.flatten()]

        if len(high_residual_points) == 0:
            return self.current_points

        # Generate new points around high residual regions
        new_points = []
        num_new = min(self.add_points, self.max_points - len(self.current_points))

        for _ in range(num_new):
            # Select a random high residual point
            idx = np.random.randint(0, len(high_residual_points))
            center = high_residual_points[idx]

            # Add noise to create new point
            noise_scale = 0.05
            new_point = center + np.random.randn(2) * noise_scale

            # Check if point is valid (in domain and outside airfoil)
            x_min, x_max = domain_bounds['x']
            y_min, y_max = domain_bounds['y']

            if (x_min <= new_point[0] <= x_max and
                    y_min <= new_point[1] <= y_max and
                    not airfoil.is_inside(new_point[0], new_point[1])):
                new_points.append(new_point)

        if len(new_points) > 0:
            new_points = np.array(new_points)
            self.current_points = np.vstack([self.current_points, new_points])

        return self.current_points


class NavierStokesPINN:
    def __init__(self, Re=200, U_inf=1.0):
        self.Re = Re
        self.U_inf = U_inf
        self.nu = U_inf * 1.0 / Re  # Kinematic viscosity (chord = 1.0)

        # Domain bounds
        self.domain_bounds = {
            'x': [-2.0, 5.0],
            'y': [-2.0, 2.0]
        }
        # Normalization parameters for inputs
        self.x_min, self.x_max = self.domain_bounds['x']
        self.y_min, self.y_max = self.domain_bounds['y']

        # Store normalization scales as tensors for GPU computation
        self.x_scale = torch.tensor([(self.x_max - self.x_min) / 2.0], device=device)
        self.x_center = torch.tensor([(self.x_max + self.x_min) / 2.0], device=device)
        self.y_scale = torch.tensor([(self.y_max - self.y_min) / 2.0], device=device)
        self.y_center = torch.tensor([(self.y_max + self.y_min) / 2.0], device=device)

        # Output normalization (velocity by U_inf, pressure by dynamic pressure)
        self.U_scale = self.U_inf
        self.p_scale = self.U_inf ** 2  # Dynamic pressure scale

        # Initialize airfoil
        self.airfoil = NACA0012()

        # Initialize network
        self.model = ModifiedMLP(
            input_dim=2,
            hidden_dim=256,  # Increased hidden dimension
            output_dim=3,
            num_layers=10,  # More layers for complex flow
            fourier_features=128,  # Doubled Fourier features
            activation='tanh'
        ).to(device)

        # Optimizer with different learning rates for different parameters
        self.optimizer = optim.Adam([
            {'params': self.model.fourier.parameters(), 'lr': 1e-4},
            {'params': self.model.layers.parameters(), 'lr': 5e-4}
        ])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=500)

        # Mixed precision training
        self.scaler = GradScaler()

        # Adaptive weighting
        self.adaptive_weight = AdaptiveWeighting(method='gradnorm')
        # Curriculum learning parameters - adaptive to total epochs
        self.total_epochs = 10000  # Default value, will be updated in train()
        self.bc_ramp_fraction = 0.9  # Use 90% of training for ramping up
        self.bc_weight_schedule = lambda epoch: min(
            1.0 + 9.0 * epoch / (self.total_epochs * self.bc_ramp_fraction),
            10.0
        )
        # Mini-batch settings for collocation points
        self.collocation_batch_size = 2500  # Use mini-batches of 2500 points
        self.full_eval_frequency = 500  # Evaluate full loss every N epochs

        self.adaptive_sampler = AdaptiveSampling(
            initial_points=7500,  # Increased from 5000
            max_points=25000,  # Increased from 15000
            add_points=300  # Increased from 500
        )

        # Generate training points
        self.generate_training_points()

        # Loss history
        self.loss_history = []
        self.periodic_loss_history = []  # Track periodic BC loss

    def normalize_inputs(self, x, y):
        """Normalize spatial coordinates to [-1, 1]"""
        x_norm = (x - self.x_center) / self.x_scale
        y_norm = (y - self.y_center) / self.y_scale
        return x_norm, y_norm

    def denormalize_inputs(self, x_norm, y_norm):
        """Denormalize from [-1, 1] back to physical coordinates"""
        x = x_norm * self.x_scale + self.x_center
        y = y_norm * self.y_scale + self.y_center
        return x, y

    def generate_training_points(self):
        """Generate all training points"""
        # Boundary points
        self.generate_boundary_points()

        # Collocation points (initial) - Store FULL pool
        collocation_points = self.adaptive_sampler.generate_initial_points(
            self.domain_bounds, self.airfoil
        )

        # Store the full pool
        self.x_collocation_full = torch.tensor(
            collocation_points[:, 0:1], dtype=torch.float32, device=device, requires_grad=True
        )
        self.y_collocation_full = torch.tensor(
            collocation_points[:, 1:2], dtype=torch.float32, device=device, requires_grad=True
        )

        # Initialize mini-batch placeholders (will be sampled in train_step)
        self.x_collocation = None
        self.y_collocation = None

    def generate_boundary_points(self):
        """Generate boundary condition points"""
        num_bc = 400  # More boundary points for better BC enforcement

        # Inlet boundary (x = -2)
        y_inlet = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], num_bc)
        x_inlet = np.ones_like(y_inlet) * self.domain_bounds['x'][0]
        self.x_inlet = torch.tensor(x_inlet.reshape(-1, 1), dtype=torch.float32, device=device)
        self.y_inlet = torch.tensor(y_inlet.reshape(-1, 1), dtype=torch.float32, device=device)

        # Outlet boundary (x = 5)
        y_outlet = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], num_bc)
        x_outlet = np.ones_like(y_outlet) * self.domain_bounds['x'][1]
        self.x_outlet = torch.tensor(x_outlet.reshape(-1, 1), dtype=torch.float32, device=device)
        self.y_outlet = torch.tensor(y_outlet.reshape(-1, 1), dtype=torch.float32, device=device)

        # Airfoil surface (no-slip)
        x_surf, y_surf = self.airfoil.surface_points()
        self.x_airfoil = torch.tensor(x_surf.reshape(-1, 1), dtype=torch.float32, device=device)
        self.y_airfoil = torch.tensor(y_surf.reshape(-1, 1), dtype=torch.float32, device=device)

        # For periodic BCs, we need matching points at top and bottom
        # Sample x locations for periodic boundary matching
        x_periodic = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], num_bc)

        # Top boundary points
        y_top = np.ones_like(x_periodic) * self.domain_bounds['y'][1]
        self.x_top = torch.tensor(x_periodic.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        self.y_top = torch.tensor(y_top.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

        # Bottom boundary points (same x locations as top for periodic matching)
        y_bottom = np.ones_like(x_periodic) * self.domain_bounds['y'][0]
        self.x_bottom = torch.tensor(x_periodic.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        self.y_bottom = torch.tensor(y_bottom.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)

    def net_NS(self, x, y):
        """Network output and derivatives for Navier-Stokes equations"""
        # Enable gradient computation for normalized coordinates
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        # Normalize inputs while maintaining gradient flow
        x_norm = (x - self.x_center) / self.x_scale
        y_norm = (y - self.y_center) / self.y_scale

        xy_norm = torch.cat([x_norm, y_norm], dim=1)
        uvp = self.model(xy_norm)

        # Keep normalized versions for gradients
        u_hat = uvp[:, 0:1]  # Normalized u
        v_hat = uvp[:, 1:2]  # Normalized v
        p_hat = uvp[:, 2:3]  # Normalized p

        # Physical values
        u = u_hat * self.U_scale
        v = v_hat * self.U_scale
        p = p_hat * self.p_scale

        # First derivatives using autograd with proper chain rule
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                  retain_graph=True, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                                  retain_graph=True, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                   retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                   retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                                   retain_graph=True, create_graph=True)[0]

        # Navier-Stokes residuals
        f_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        f_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        f_cont = u_x + v_y

        return u, v, p, f_u, f_v, f_cont

    def compute_losses(self, use_full_batch=False):
        """Compute all loss terms with periodic boundary conditions"""
        losses = {}

        # Use full batch for evaluation or mini-batch for training
        if use_full_batch:
            x_col = self.x_collocation_full
            y_col = self.y_collocation_full
        else:
            x_col = self.x_collocation
            y_col = self.y_collocation

        # Physics loss (Navier-Stokes equations)
        u_pred, v_pred, p_pred, f_u, f_v, f_cont = self.net_NS(x_col, y_col)
        losses['physics_u'] = torch.mean(f_u ** 2)
        losses['physics_v'] = torch.mean(f_v ** 2)
        losses['continuity'] = torch.mean(f_cont ** 2)

        # Boundary conditions (work with normalized coordinates and outputs)

        # Inlet (normalized)
        x_inlet_norm, y_inlet_norm = self.normalize_inputs(self.x_inlet, self.y_inlet)
        xy_inlet_norm = torch.cat([x_inlet_norm, y_inlet_norm], dim=1)
        uvp_inlet = self.model(xy_inlet_norm)  # Returns u/U_inf, v/U_inf, p/U_inf^2
        losses['bc_inlet_u'] = torch.mean((uvp_inlet[:, 0:1] - 1.0) ** 2)  # u/U_inf = 1
        losses['bc_inlet_v'] = torch.mean(uvp_inlet[:, 1:2] ** 2)  # v/U_inf = 0

        # Outlet (normalized)
        x_outlet_norm, y_outlet_norm = self.normalize_inputs(self.x_outlet, self.y_outlet)
        xy_outlet_norm = torch.cat([x_outlet_norm, y_outlet_norm], dim=1)
        uvp_outlet = self.model(xy_outlet_norm)
        losses['bc_outlet_p'] = torch.mean(uvp_outlet[:, 2:3] ** 2)  # p/U_inf^2 = 0

        # Airfoil (no-slip, normalized)
        # Get current epoch for curriculum learning (add epoch parameter to compute_losses)
        bc_weight = self.bc_weight_schedule(self.current_epoch) if hasattr(self, 'current_epoch') else 10.0

        # Airfoil (no-slip, normalized) with curriculum learning
        x_airfoil_norm, y_airfoil_norm = self.normalize_inputs(self.x_airfoil, self.y_airfoil)
        xy_airfoil_norm = torch.cat([x_airfoil_norm, y_airfoil_norm], dim=1)
        uvp_airfoil = self.model(xy_airfoil_norm)
        losses['bc_airfoil_u'] = torch.mean(uvp_airfoil[:, 0:1] ** 2) * bc_weight
        losses['bc_airfoil_v'] = torch.mean(uvp_airfoil[:, 1:2] ** 2) * bc_weight

        # PERIODIC BOUNDARY CONDITIONS at top and bottom
        # Compute flow at top boundary
        x_top_norm, y_top_norm = self.normalize_inputs(self.x_top, self.y_top)
        xy_top_norm = torch.cat([x_top_norm, y_top_norm], dim=1)

        # Enable gradients for derivative computation
        xy_top_norm.requires_grad_(True)
        uvp_top = self.model(xy_top_norm)
        u_top = uvp_top[:, 0:1] * self.U_scale
        v_top = uvp_top[:, 1:2] * self.U_scale
        p_top = uvp_top[:, 2:3] * self.p_scale

        # Compute flow at bottom boundary
        x_bottom_norm, y_bottom_norm = self.normalize_inputs(self.x_bottom, self.y_bottom)
        xy_bottom_norm = torch.cat([x_bottom_norm, y_bottom_norm], dim=1)

        xy_bottom_norm.requires_grad_(True)
        uvp_bottom = self.model(xy_bottom_norm)
        u_bottom = uvp_bottom[:, 0:1] * self.U_scale
        v_bottom = uvp_bottom[:, 1:2] * self.U_scale
        p_bottom = uvp_bottom[:, 2:3] * self.p_scale

        # Periodic BC: u(x, y_top) = u(x, y_bottom), same for v and p
        periodic_weight = 5.0  # Weight for periodic BC enforcement
        losses['bc_periodic_u'] = torch.mean((u_top - u_bottom) ** 2) * periodic_weight
        losses['bc_periodic_v'] = torch.mean((v_top - v_bottom) ** 2) * periodic_weight
        losses['bc_periodic_p'] = torch.mean((p_top - p_bottom) ** 2) * periodic_weight

        # Also enforce derivative continuity for smoother periodic transition
        # Compute y-derivatives at boundaries
        u_y_top = torch.autograd.grad(u_top.sum(), self.y_top,
                                      retain_graph=True, create_graph=True)[0]
        u_y_bottom = torch.autograd.grad(u_bottom.sum(), self.y_bottom,
                                         retain_graph=True, create_graph=True)[0]
        v_y_top = torch.autograd.grad(v_top.sum(), self.y_top,
                                      retain_graph=True, create_graph=True)[0]
        v_y_bottom = torch.autograd.grad(v_bottom.sum(), self.y_bottom,
                                         retain_graph=True, create_graph=True)[0]

        # Periodic derivative matching
        derivative_weight = 2.0
        losses['bc_periodic_du_dy'] = torch.mean((u_y_top - u_y_bottom) ** 2) * derivative_weight
        losses['bc_periodic_dv_dy'] = torch.mean((v_y_top - v_y_bottom) ** 2) * derivative_weight

        return losses, f_u ** 2 + f_v ** 2 + f_cont ** 2  # Return residuals for adaptive sampling

    def lbfgs_closure(self):
        """Closure function for L-BFGS optimizer"""
        self.lbfgs_optimizer.zero_grad()

        # Compute losses on full batch for L-BFGS
        losses, _ = self.compute_losses(use_full_batch=True)

        # Use fixed weights for L-BFGS (from final Adam training)
        if hasattr(self, 'final_weights'):
            weights = self.final_weights
        else:
            weights = {k: 1.0 for k in losses.keys()}

        # Weighted total loss
        total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())

        # Backward pass
        total_loss.backward()

        # Store loss for monitoring
        self.lbfgs_loss = total_loss.item()
        self.lbfgs_losses = {k: v.item() for k, v in losses.items()}

        return total_loss

    def train_lbfgs(self, max_iter=500, print_every=10):
        """Fine-tune with L-BFGS optimizer"""
        print("\n" + "=" * 60)
        print("Starting L-BFGS fine-tuning...")
        print("=" * 60)

        # Store final weights from Adam training
        if hasattr(self.adaptive_weight, 'weights'):
            self.final_weights = self.adaptive_weight.weights.copy()
        else:
            self.final_weights = {k: 1.0 for k in ['physics_u', 'physics_v', 'continuity',
                                                   'bc_inlet_u', 'bc_inlet_v', 'bc_outlet_p',
                                                   'bc_airfoil_u', 'bc_airfoil_v',
                                                   'bc_periodic_u', 'bc_periodic_v', 'bc_periodic_p',
                                                   'bc_periodic_du_dy', 'bc_periodic_dv_dy']}

        # Create L-BFGS optimizer
        self.lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=1.0,  # L-BFGS learning rate
            max_iter=20,  # Number of iterations per step
            max_eval=25,  # Max function evaluations per step
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,  # Increase history for better approximation
            line_search_fn='strong_wolfe'
        )

        # Set model to train mode
        self.model.train()

        # Store current epoch for BC weight
        self.current_epoch = self.total_epochs  # Use maximum BC weight

        start_time = time.time()

        for iteration in range(max_iter):
            # Perform L-BFGS step
            self.lbfgs_optimizer.step(self.lbfgs_closure)

            # Add to loss history
            self.loss_history.append(self.lbfgs_loss)

            # Print progress
            if iteration % print_every == 0:
                elapsed = time.time() - start_time
                forces = self.compute_forces()
                print(f"L-BFGS Iter {iteration}/{max_iter}, Loss: {self.lbfgs_loss:.6f}, "
                      f"CD: {forces['CD_total']:.6f}, Time: {elapsed:.2f}s")
                print(f"  Periodic BC losses: u={self.lbfgs_losses.get('bc_periodic_u', 0):.6f}, "
                      f"v={self.lbfgs_losses.get('bc_periodic_v', 0):.6f}, "
                      f"p={self.lbfgs_losses.get('bc_periodic_p', 0):.6f}")

                # Check convergence
                if iteration > 50 and len(self.loss_history) > 50:
                    recent_losses = self.loss_history[-50:]
                    loss_std = np.std(recent_losses)
                    loss_mean = np.mean(recent_losses)
                    if loss_std / loss_mean < 1e-5:  # Relative change is very small
                        print(f"  Converged (relative change < 1e-5)")
                        break

        print(f"\nL-BFGS fine-tuning completed in {time.time() - start_time:.2f} seconds")

        # Final evaluation
        forces = self.compute_forces()
        print(f"Final CD after L-BFGS: {forces['CD_total']:.6f}")

    def train_step(self, epoch):
        """Single training step"""
        self.model.train()

        # Store current epoch for curriculum learning
        self.current_epoch = epoch

        # Sample mini-batch from full collocation pool
        num_collocation_full = len(self.x_collocation_full)
        batch_size = min(self.collocation_batch_size, num_collocation_full)
        indices = torch.randperm(num_collocation_full)[:batch_size]

        self.x_collocation = self.x_collocation_full[indices]
        self.y_collocation = self.y_collocation_full[indices]

        # Decide whether to use full batch for evaluation
        use_full_batch = (epoch % self.full_eval_frequency == 0) and (epoch > 0)

        # Forward pass with mixed precision
        with autocast():
            losses, residuals = self.compute_losses(use_full_batch=use_full_batch)

        # Track periodic loss
        periodic_loss = (losses.get('bc_periodic_u', 0) +
                         losses.get('bc_periodic_v', 0) +
                         losses.get('bc_periodic_p', 0))
        self.periodic_loss_history.append(periodic_loss.item() if torch.is_tensor(periodic_loss) else periodic_loss)

        # Compute adaptive weights
        if epoch > 100:  # Start adaptive weighting after initial training
            weights = self.adaptive_weight.compute_weights(losses)
        else:
            weights = {k: 1.0 for k in losses.keys()}

        # Weighted total loss
        total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Adaptive sampling (every 200 epochs) - use full batch for residual evaluation
        if epoch > 0 and epoch % 200 == 0:
            # Compute residuals on full pool for adaptive sampling
            with autocast():
                _, _, _, f_u_full, f_v_full, f_cont_full = self.net_NS(
                    self.x_collocation_full, self.y_collocation_full
                )
                # Detach residuals before using for adaptive sampling
                residuals_full = (f_u_full ** 2 + f_v_full ** 2 + f_cont_full ** 2).detach()

            new_points = self.adaptive_sampler.add_new_points(
                residuals_full, self.domain_bounds, self.airfoil
            )
            if len(new_points) > len(self.x_collocation_full):
                # Update the full pool
                self.x_collocation_full = torch.tensor(
                    new_points[:, 0:1], dtype=torch.float32, device=device, requires_grad=True
                )
                self.y_collocation_full = torch.tensor(
                    new_points[:, 1:2], dtype=torch.float32, device=device, requires_grad=True
                )
                print(f"Epoch {epoch}: Added points. Total collocation points: {len(new_points)}")
                print(f"         Using mini-batches of size: {self.collocation_batch_size}")

        return total_loss.item(), {k: v.item() for k, v in losses.items()}

    def evaluate_full_loss(self):
        """Evaluate loss on full collocation pool"""
        self.model.eval()
        # Note: We need gradients for net_NS, but we'll detach the final loss
        with autocast():
            losses, _ = self.compute_losses(use_full_batch=True)

        # Detach losses to prevent gradient accumulation
        total_loss = sum(v.detach() for v in losses.values())
        return total_loss.item(), {k: v.detach().item() for k, v in losses.items()}

    def train(self, epochs=10000, print_every=100):
        """Training loop"""
        # Update total epochs for adaptive curriculum
        self.total_epochs = epochs
        # Recreate the schedule with the new total epochs
        self.bc_weight_schedule = lambda epoch: min(
            1.0 + 9.0 * epoch / (self.total_epochs * self.bc_ramp_fraction),
            10.0
        )

        print(f"Starting training for {epochs} epochs with PERIODIC boundary conditions...")
        print(f"Domain: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]")
        print(f"Solidity = chord/domain_height = 1.0/4.0 = 0.25")
        print(f"BC weight will reach maximum at epoch {int(self.total_epochs * self.bc_ramp_fraction)}")
        start_time = time.time()

        for epoch in range(epochs):
            loss, loss_components = self.train_step(epoch)
            self.loss_history.append(loss)

            # Learning rate scheduling
            self.scheduler.step(loss)

            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                # Check if this was a full evaluation epoch
                if epoch % self.full_eval_frequency == 0 and epoch > 0:
                    full_loss, full_components = self.evaluate_full_loss()
                    forces = self.compute_forces()
                    print(
                        f"Epoch {epoch}/{epochs}, Mini-batch Loss: {loss:.6f}, Full Loss: {full_loss:.6f}, "
                        f"CD: {forces['CD_total']:.6f}, Time: {elapsed:.2f}s")
                    print(f"  Periodic BC: u={full_components.get('bc_periodic_u', 0):.6f}, "
                          f"v={full_components.get('bc_periodic_v', 0):.6f}, "
                          f"p={full_components.get('bc_periodic_p', 0):.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}, Time: {elapsed:.2f}s")
                    print(f"  Periodic BC: u={loss_components.get('bc_periodic_u', 0):.6f}, "
                          f"v={loss_components.get('bc_periodic_v', 0):.6f}, "
                          f"p={loss_components.get('bc_periodic_p', 0):.6f}")

        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def continue_training(self, additional_epochs=5000, print_every=100):
        """Continue training from current state"""
        start_epoch = len(self.loss_history)
        end_epoch = start_epoch + additional_epochs

        # For continuation, keep BC weight at maximum
        self.bc_weight_schedule = lambda epoch: 10.0

        print(f"Continuing training from epoch {start_epoch} to {end_epoch}...")
        start_time = time.time()

        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            loss, loss_components = self.train_step(epoch)
            self.loss_history.append(loss)

            # Learning rate scheduling
            self.scheduler.step(loss)

            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{end_epoch}, Loss: {loss:.6f}, Time: {elapsed:.2f}s")
                print(f"  Components: {loss_components}")

        print(f"Continuation training completed in {time.time() - start_time:.2f} seconds")

    def predict(self, x, y):
        """Predict flow field at given points"""
        self.model.eval()
        x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)

        # Normalize inputs
        x_norm, y_norm = self.normalize_inputs(x_tensor, y_tensor)
        xy_norm = torch.cat([x_norm, y_norm], dim=1)

        with torch.no_grad():
            uvp = self.model(xy_norm)  # Returns normalized values

        # Denormalize outputs
        u = (uvp[:, 0] * self.U_scale).cpu().numpy().reshape(x.shape)
        v = (uvp[:, 1] * self.U_scale).cpu().numpy().reshape(x.shape)
        p = (uvp[:, 2] * self.p_scale).cpu().numpy().reshape(x.shape)

        return u, v, p

    def compute_forces(self):
        """Compute lift and drag coefficients"""
        # Get airfoil surface points
        x_surf, y_surf = self.airfoil.surface_points()
        n_points = len(x_surf)

        # Predict on surface
        self.model.eval()
        x_tensor = torch.tensor(x_surf.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        y_tensor = torch.tensor(y_surf.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        # Normalize coordinates
        x_norm, y_norm = self.normalize_inputs(x_tensor, y_tensor)
        xy_norm = torch.cat([x_norm, y_norm], dim=1)

        # Need to temporarily set model to train mode for gradients
        self.model.train()

        uvp = self.model(xy_norm)  # Returns normalized values

        # Extract and denormalize pressure
        p = uvp[:, 2:3] * self.p_scale  # Denormalize pressure

        # Compute velocity gradients with proper chain rule for normalized coordinates
        u_hat = uvp[:, 0:1]  # Normalized u
        v_hat = uvp[:, 1:2]  # Normalized v

        u_hat_x_norm = torch.autograd.grad(u_hat, x_norm, grad_outputs=torch.ones_like(u_hat),
                                           retain_graph=True, create_graph=False)[0]
        u_hat_y_norm = torch.autograd.grad(u_hat, y_norm, grad_outputs=torch.ones_like(u_hat),
                                           retain_graph=True, create_graph=False)[0]
        v_hat_x_norm = torch.autograd.grad(v_hat, x_norm, grad_outputs=torch.ones_like(v_hat),
                                           retain_graph=True, create_graph=False)[0]
        v_hat_y_norm = torch.autograd.grad(v_hat, y_norm, grad_outputs=torch.ones_like(v_hat),
                                           retain_graph=True, create_graph=False)[0]

        # Apply chain rule
        u_x = u_hat_x_norm * self.U_scale / self.x_scale
        u_y = u_hat_y_norm * self.U_scale / self.y_scale
        v_x = v_hat_x_norm * self.U_scale / self.x_scale
        v_y = v_hat_y_norm * self.U_scale / self.y_scale

        # Set back to eval mode
        self.model.eval()

        # Convert to numpy
        p_surf = p.detach().cpu().numpy().flatten()
        u_x_surf = u_x.detach().cpu().numpy().flatten()
        u_y_surf = u_y.detach().cpu().numpy().flatten()
        v_x_surf = v_x.detach().cpu().numpy().flatten()
        v_y_surf = v_y.detach().cpu().numpy().flatten()

        # Compute normals (pointing outward from airfoil)
        normals = np.zeros((n_points, 2))
        for i in range(n_points):
            i_prev = (i - 1) % n_points
            i_next = (i + 1) % n_points

            # Tangent vector
            dx = x_surf[i_next] - x_surf[i_prev]
            dy = y_surf[i_next] - y_surf[i_prev]

            # Normal (rotate tangent by 90 degrees)
            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm > 1e-10:  # Avoid division by zero
                normals[i, 0] = -dy / norm
                normals[i, 1] = dx / norm
            else:
                normals[i, 0] = 0
                normals[i, 1] = 1

        # Ensure normals point outward
        for i in range(n_points):
            # Check if normal points outward (dot product with position vector should be positive)
            if x_surf[i] * normals[i, 0] + y_surf[i] * normals[i, 1] < 0:
                normals[i] *= -1

        # Compute forces using trapezoidal integration
        Fx_pressure = 0
        Fy_pressure = 0
        Fx_viscous = 0
        Fy_viscous = 0

        for i in range(n_points):
            i_next = (i + 1) % n_points

            # Arc length
            ds = np.sqrt((x_surf[i_next] - x_surf[i]) ** 2 +
                         (y_surf[i_next] - y_surf[i]) ** 2)

            if ds < 1e-10:  # Skip if points are too close
                continue

            # Average values at midpoint
            p_avg = 0.5 * (p_surf[i] + p_surf[i_next] if i_next < len(p_surf) else p_surf[i])
            normal_avg = 0.5 * (normals[i] + normals[i_next % n_points])

            # Pressure forces (pressure acts normal to surface)
            Fx_pressure -= p_avg * normal_avg[0] * ds
            Fy_pressure -= p_avg * normal_avg[1] * ds

            # Viscous stress tensor components (average) - corrected formulation
            tau_xx = 2.0 * self.nu * 0.5 * (u_x_surf[i] + u_x_surf[i_next % len(u_x_surf)])
            tau_yy = 2.0 * self.nu * 0.5 * (v_y_surf[i] + v_y_surf[i_next % len(v_y_surf)])
            tau_xy = 0.5 * self.nu * ((u_y_surf[i] + v_x_surf[i]) +
                                      (u_y_surf[i_next % len(u_y_surf)] +
                                       v_x_surf[i_next % len(v_x_surf)]))

            # Viscous forces
            Fx_viscous += (tau_xx * normal_avg[0] + tau_xy * normal_avg[1]) * ds
            Fy_viscous += (tau_xy * normal_avg[0] + tau_yy * normal_avg[1]) * ds

        # Non-dimensionalize
        q_inf = 0.5 * self.U_inf ** 2  # Dynamic pressure
        chord = 1.0

        # Check for invalid values
        if np.isnan(Fx_pressure) or np.isnan(Fy_pressure) or np.isnan(Fx_viscous) or np.isnan(Fy_viscous):
            print("Warning: NaN detected in force calculations")
            print(f"Fx_pressure: {Fx_pressure}, Fy_pressure: {Fy_pressure}")
            print(f"Fx_viscous: {Fx_viscous}, Fy_viscous: {Fy_viscous}")
            return {
                'CD_total': 0.0,
                'CL_total': 0.0,
                'CD_pressure': 0.0,
                'CD_viscous': 0.0,
                'CL_pressure': 0.0,
                'CL_viscous': 0.0
            }

        CD_pressure = Fx_pressure / (q_inf * chord)
        CL_pressure = Fy_pressure / (q_inf * chord)
        CD_viscous = Fx_viscous / (q_inf * chord)
        CL_viscous = Fy_viscous / (q_inf * chord)

        CD_total = CD_pressure + CD_viscous
        CL_total = CL_pressure + CL_viscous

        return {
            'CD_total': float(CD_total),
            'CL_total': float(CL_total),
            'CD_pressure': float(CD_pressure),
            'CD_viscous': float(CD_viscous),
            'CL_pressure': float(CL_pressure),
            'CL_viscous': float(CL_viscous)
        }

    def compute_Cp(self):
        """Compute pressure coefficient on airfoil surface"""
        x_surf, y_surf = self.airfoil.surface_points()

        # Predict pressure on surface
        _, _, p_surf = self.predict(x_surf, y_surf)

        # Compute Cp
        q_inf = 0.5 * self.U_inf ** 2
        Cp = (p_surf - 0) / q_inf  # p_inf = 0 at outlet

        return x_surf, y_surf, Cp

    def plot_results(self):
        """Generate all plots with periodic BC visualization"""
        # Create grid for plotting
        nx, ny = 200, 100
        x = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], nx)
        y = np.linspace(self.domain_bounds['y'][0], self.domain_bounds['y'][1], ny)
        X, Y = np.meshgrid(x, y)

        # Mask points inside airfoil
        mask = np.zeros_like(X, dtype=bool)
        for i in range(ny):
            for j in range(nx):
                if self.airfoil.is_inside(X[i, j], Y[i, j]):
                    mask[i, j] = True

        # Predict flow field
        print("Generating flow field predictions...")
        U, V, P = self.predict(X, Y)

        # Mask velocities inside airfoil
        U[mask] = 0
        V[mask] = 0
        P[mask] = np.nan

        # Velocity magnitude
        U_mag = np.sqrt(U ** 2 + V ** 2)
        U_mag[mask] = np.nan

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Velocity magnitude contour
        ax1 = plt.subplot(3, 3, 1)
        contour = ax1.contourf(X, Y, U_mag, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax1, label='Velocity Magnitude')
        x_surf, y_surf = self.airfoil.surface_points()
        ax1.fill(x_surf, y_surf, 'white', zorder=2)
        ax1.plot(x_surf, y_surf, 'k-', linewidth=2)
        ax1.set_xlim([-0.5, 2])
        ax1.set_ylim([-1, 1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Velocity Magnitude')
        ax1.set_aspect('equal')

        # 2. Pressure contour
        ax2 = plt.subplot(3, 3, 2)
        contour = ax2.contourf(X, Y, P, levels=50, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax2, label='Pressure')
        ax2.fill(x_surf, y_surf, 'white', zorder=2)
        ax2.plot(x_surf, y_surf, 'k-', linewidth=2)
        ax2.set_xlim([-0.5, 2])
        ax2.set_ylim([-1, 1])
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Pressure Field')
        ax2.set_aspect('equal')

        # 3. Streamlines
        ax3 = plt.subplot(3, 3, 3)
        # Create finer grid for streamlines
        strm = ax3.streamplot(X, Y, U, V, density=2, linewidth=1,
                              arrowsize=1.5, color=U_mag, cmap='plasma')
        plt.colorbar(strm.lines, ax=ax3, label='Velocity Magnitude')
        ax3.fill(x_surf, y_surf, 'white', zorder=2)
        ax3.plot(x_surf, y_surf, 'k-', linewidth=2)
        ax3.set_xlim([-0.5, 2])
        ax3.set_ylim([-1, 1])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Streamlines')
        ax3.set_aspect('equal')

        # 4. Vorticity
        ax4 = plt.subplot(3, 3, 4)
        # Compute vorticity
        dv_dx = np.gradient(V, x, axis=1)
        du_dy = np.gradient(U, y, axis=0)
        vorticity = dv_dx - du_dy
        vorticity[mask] = np.nan
        contour = ax4.contourf(X, Y, vorticity, levels=50, cmap='seismic',
                               vmin=-10, vmax=10)
        plt.colorbar(contour, ax=ax4, label='Vorticity')
        ax4.fill(x_surf, y_surf, 'white', zorder=2)
        ax4.plot(x_surf, y_surf, 'k-', linewidth=2)
        ax4.set_xlim([-0.5, 2])
        ax4.set_ylim([-1, 1])
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_title('Vorticity')
        ax4.set_aspect('equal')

        # 5. Cp distribution
        ax5 = plt.subplot(3, 3, 5)
        x_cp, y_cp, Cp = self.compute_Cp()

        # The surface_points method returns points in a specific order:
        # First half: upper surface from LE to TE
        # Second half: lower surface from TE to LE (reversed)
        n_surf = len(x_cp)
        n_half = n_surf // 2

        # Upper surface (first half of points)
        x_upper = x_cp[:n_half]
        Cp_upper = Cp[:n_half]

        # Lower surface (second half of points, needs to be reversed for proper plotting)
        x_lower = x_cp[n_half:]
        Cp_lower = Cp[n_half:]

        # Sort both surfaces by x-coordinate to ensure proper plotting
        upper_sort = np.argsort(x_upper)
        lower_sort = np.argsort(x_lower)

        ax5.plot(x_upper[upper_sort], -Cp_upper[upper_sort], 'b-', label='Upper surface', linewidth=2)
        ax5.plot(x_lower[lower_sort], -Cp_lower[lower_sort], 'r-', label='Lower surface', linewidth=2)
        ax5.set_xlabel('x/c')
        ax5.set_ylabel('-Cp')
        ax5.set_title('Pressure Coefficient Distribution')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.invert_yaxis()

        # 6. Wake velocity profile
        ax6 = plt.subplot(3, 3, 6)
        # Extract wake profile at x = 1.5 (downstream)
        x_wake = 1.5
        y_wake = np.linspace(-1, 1, 100)
        X_wake = np.ones_like(y_wake) * x_wake
        U_wake, _, _ = self.predict(X_wake, y_wake)
        ax6.plot(U_wake / self.U_inf, y_wake, 'b-', linewidth=2)
        ax6.set_xlabel('u/U∞')
        ax6.set_ylabel('y')
        ax6.set_title(f'Wake Velocity Profile at x = {x_wake}')
        ax6.grid(True, alpha=0.3)
        ax6.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)

        # 7. Force coefficients
        ax7 = plt.subplot(3, 3, 7)
        forces = self.compute_forces()
        categories = ['CD\n(total)', 'CD\n(pressure)', 'CD\n(viscous)',
                      'CL\n(total)', 'CL\n(pressure)', 'CL\n(viscous)']
        values = [forces['CD_total'], forces['CD_pressure'], forces['CD_viscous'],
                  forces['CL_total'], forces['CL_pressure'], forces['CL_viscous']]
        colors = ['red', 'orange', 'yellow', 'blue', 'cyan', 'lightblue']
        bars = ax7.bar(categories, values, color=colors)
        ax7.set_ylabel('Coefficient Value')
        ax7.set_title('Force Coefficients')
        ax7.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.4f}', ha='center', va='bottom' if val > 0 else 'top')

        # 8. Training loss
        ax8 = plt.subplot(3, 3, 8)
        ax8.semilogy(self.loss_history, 'b-', linewidth=2)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Total Loss')
        ax8.set_title('Training History')
        ax8.grid(True, alpha=0.3)

        # 9. Periodic BC verification - Compare top and bottom boundary profiles
        ax9 = plt.subplot(3, 3, 9)
        # Sample multiple x locations
        x_samples = np.linspace(self.domain_bounds['x'][0], self.domain_bounds['x'][1], 50)

        # Get velocities at top and bottom boundaries
        y_top = np.ones_like(x_samples) * self.domain_bounds['y'][1]
        y_bottom = np.ones_like(x_samples) * self.domain_bounds['y'][0]

        U_top, V_top, P_top = self.predict(x_samples, y_top)
        U_bottom, V_bottom, P_bottom = self.predict(x_samples, y_bottom)

        # Plot comparison
        ax9.plot(x_samples, U_top, 'b-', label='u (top)', linewidth=2)
        ax9.plot(x_samples, U_bottom, 'b--', label='u (bottom)', linewidth=2)
        ax9.plot(x_samples, V_top, 'r-', label='v (top)', linewidth=2)
        ax9.plot(x_samples, V_bottom, 'r--', label='v (bottom)', linewidth=2)

        ax9.set_xlabel('x')
        ax9.set_ylabel('Velocity')
        ax9.set_title('Periodic BC Verification (Top vs Bottom)')
        ax9.grid(True, alpha=0.3)
        ax9.legend(loc='best', fontsize=8)

        # Add text showing the maximum difference
        max_diff_u = np.max(np.abs(U_top - U_bottom))
        max_diff_v = np.max(np.abs(V_top - V_bottom))
        ax9.text(0.02, 0.98, f'Max diff: u={max_diff_u:.4f}, v={max_diff_v:.4f}',
                 transform=ax9.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('naca0012_periodic_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print force coefficients and periodic BC error
        print("\n" + "=" * 50)
        print("FORCE COEFFICIENTS:")
        print("=" * 50)
        print(f"Total Drag Coefficient (CD):     {forces['CD_total']:.6f}")
        print(f"  - Pressure contribution:        {forces['CD_pressure']:.6f}")
        print(f"  - Viscous contribution:         {forces['CD_viscous']:.6f}")
        print(f"Total Lift Coefficient (CL):     {forces['CL_total']:.6f}")
        print(f"  - Pressure contribution:        {forces['CL_pressure']:.6f}")
        print(f"  - Viscous contribution:         {forces['CL_viscous']:.6f}")
        print("=" * 50)
        print("PERIODIC BC VERIFICATION:")
        print("=" * 50)
        print(f"Max difference in u (top vs bottom): {max_diff_u:.6f}")
        print(f"Max difference in v (top vs bottom): {max_diff_v:.6f}")
        print(f"Max difference in p (top vs bottom): {np.max(np.abs(P_top - P_bottom)):.6f}")
        print("=" * 50)


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize PINN
    pinn = NavierStokesPINN(Re=200, U_inf=1.0)

    # Train the model with Adam
    pinn.train(epochs=8000, print_every=500)

    # Fine-tune with L-BFGS for better accuracy
    pinn.train_lbfgs(max_iter=300, print_every=20)

    # Generate plots and results
    pinn.plot_results()

    # Save the trained model
    torch.save({
        'model_state_dict': pinn.model.state_dict(),
        'optimizer_state_dict': pinn.optimizer.state_dict(),
        'loss_history': pinn.loss_history,
        'periodic_loss_history': pinn.periodic_loss_history,
    }, 'naca0012_periodic_pinn_model.pth')
    print("\nModel saved to 'naca0012_periodic_pinn_model.pth'")


if __name__ == "__main__":
    main()
