import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

class SceneSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        
        # Keep track of the center for the whole scene
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), 
            requires_grad=cfg.center.opt
        )
        
        # Smoothing factor for smooth min operations
        self.smooth_factor = cfg.smooth_factor if hasattr(cfg, 'smooth_factor') else 0.2
        
        # Create a list of primitives
        self.primitives = torch.nn.ModuleList()
        
        for i, prim_cfg in enumerate(cfg.primitives):
            # Create the primitive based on its type
            prim_type = prim_cfg.type
            primitive = sdf_dict[prim_type](prim_cfg)
            self.primitives.append(primitive)
    
    def smooth_min(self, a, b, k=0.2):
        """
        Smooth minimum function for creating smooth unions between SDF primitives
        """
        h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return a * h + b * (1.0 - h) - k * h * (1.0 - h)
        
    def forward(self, points):
        points = points.view(-1, 3)
        
        # Initialize with a large positive value
        min_sdf = torch.ones_like(points[:, 0:1]) * 1000.0
        
        # Compute the minimum SDF across all primitives
        for primitive in self.primitives:
            sdf = primitive(points)
            min_sdf = self.smooth_min(min_sdf, sdf, self.smooth_factor)
            
        return min_sdf
    
sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
    'scene': SceneSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        self.layer1 = torch.nn.Linear(embedding_dim_xyz, cfg.n_hidden_neurons_xyz) 
        self.layer2 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer3 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer4 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer5 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        
        self.layer6 = torch.nn.Linear(cfg.n_hidden_neurons_xyz + embedding_dim_xyz, cfg.n_hidden_neurons_xyz)
        self.layer7 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer8 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        
        self.dropout = torch.nn.Dropout(0.1) if hasattr(cfg, 'use_dropout') and cfg.use_dropout else None
        self.density_output = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)  # Output for density
        self.bottleneck = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir)  # Bottleneck for color
        
        self.color_layer1 = torch.nn.Linear(cfg.n_hidden_neurons_dir + embedding_dim_dir, cfg.n_hidden_neurons_dir)
        self.color_output = torch.nn.Linear(cfg.n_hidden_neurons_dir, 3)  # RGB output
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points
        batch_size, n_samples, _ = sample_points.shape
        flat_samples = sample_points.reshape(-1, 3)
        directions = ray_bundle.directions
        directions = torch.nn.functional.normalize(directions, dim=-1)
        sample_dirs = directions.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)
        pos_embeddings = self.harmonic_embedding_xyz(flat_samples)
        dir_embeddings = self.harmonic_embedding_dir(sample_dirs)
        
        x = self.relu(self.layer1(pos_embeddings))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        
        x = torch.cat([x, pos_embeddings], dim=-1)
        
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        density = self.relu(self.density_output(x))
        bottleneck_features = self.relu(self.bottleneck(x))
        color_input = torch.cat([bottleneck_features, dir_embeddings], dim=-1)
        color_features = self.relu(self.color_layer1(color_input))
        
        if hasattr(self, 'dropout'):
           color_features = self.dropout(color_features)
           
        color = self.sigmoid(self.color_output(color_features))
        density = density.reshape(batch_size, n_samples, 1)
        color = color.reshape(batch_size, n_samples, 3)
        
        return {
            'density': density,
            'feature': color
        }

class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        
        self.layer1 = torch.nn.Linear(embedding_dim_xyz, cfg.n_hidden_neurons_xyz)
        self.layer2 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer3 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer4 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        
        self.layer5 = torch.nn.Linear(cfg.n_hidden_neurons_xyz + embedding_dim_xyz, cfg.n_hidden_neurons_xyz)
        self.layer6 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        self.layer7 = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz)
        
        self.sdf_output = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)
        
        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.feature_layer = torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir)
        self.color_layer1 = torch.nn.Linear(cfg.n_hidden_neurons_dir, cfg.n_hidden_neurons_dir)
        self.color_output = torch.nn.Linear(cfg.n_hidden_neurons_dir, 3)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        
    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        embedded_points = self.harmonic_embedding_xyz(points)
        
        x = self.relu(self.layer1(embedded_points))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        
        x = torch.cat([x, embedded_points], dim=-1)
        
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        
        sdf = self.sdf_output(x)
        return sdf
        
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        embedded_points = self.harmonic_embedding_xyz(points)
        
        x = self.relu(self.layer1(embedded_points))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        
        x = torch.cat([x, embedded_points], dim=-1)
        
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        
        features = self.relu(self.feature_layer(x))
        color_features = self.relu(self.color_layer1(features))
        color = self.sigmoid(self.color_output(color_features))
        
        return color
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)
        embedded_points = self.harmonic_embedding_xyz(points)
        
        x = self.relu(self.layer1(embedded_points))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        
        x = torch.cat([x, embedded_points], dim=-1)
        
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        
        sdf = self.sdf_output(x)
        
        features = self.relu(self.feature_layer(x))
        color_features = self.relu(self.color_layer1(features))
        color = self.sigmoid(self.color_output(color_features))
        
        return sdf, color
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        points = points.view(-1, 3)
        points.requires_grad_(True)
    
        # Compute SDF
        sdf = self.get_distance(points)
    
        # Compute gradient using autograd
        grad_outputs = torch.ones_like(sdf, device=points.device)
        gradients = torch.autograd.grad(
          outputs=sdf,
          inputs=points,
          grad_outputs=grad_outputs,
          create_graph=True,
          retain_graph=True,
          only_inputs=True
        )[0]
    
        return sdf, gradients

class HierarchicalSampler(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.n_coarse_samples = cfg.n_coarse_samples
        self.n_fine_samples = cfg.n_fine_samples
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.training = True
        
    def sample_coarse(self, ray_bundle):
        batch_size = ray_bundle.origins.shape[0]
        
        z_vals = torch.linspace(
            self.min_depth, 
            self.max_depth, 
            self.n_coarse_samples, 
            device=ray_bundle.origins.device
        ).expand(batch_size, self.n_coarse_samples)
        
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        
        origins = ray_bundle.origins.unsqueeze(1).expand(-1, self.n_coarse_samples, -1)
        directions = ray_bundle.directions.unsqueeze(1).expand(-1, self.n_coarse_samples, -1)
        sample_points = origins + directions * z_vals.unsqueeze(-1)
        
        coarse_bundle = ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(-1)
        )
        
        return coarse_bundle, z_vals

    def sample_fine(self, ray_bundle, z_vals, weights):
        batch_size = ray_bundle.origins.shape[0]
        
        weights = weights + 1e-5 
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        if self.training:
            u = torch.rand(batch_size, self.n_fine_samples, device=weights.device)
        else:
            u = torch.linspace(0., 1., steps=self.n_fine_samples, device=weights.device)
            u = u.expand(batch_size, self.n_fine_samples)
            u = u + torch.rand(batch_size, 1, device=weights.device) * (1./self.n_fine_samples)
        
        u = u.contiguous()
        z_edges = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_edges = torch.cat([
          z_vals[..., :1] - 0.5 * (z_vals[..., 1:2] - z_vals[..., :1]),
          z_edges,
          z_vals[..., -1:] + 0.5 * (z_vals[..., -1:] - z_vals[..., -2:-1])
        ], dim=-1)
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds-1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1]-1)
        
        inds_g = torch.stack([below, above], dim=-1)
        cdf_shape = cdf.shape
        # matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(batch_size, self.n_fine_samples, cdf_shape[-1]), 2, inds_g)
        # bins_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)
        edges_g = torch.gather(z_edges.unsqueeze(1).expand(batch_size, self.n_fine_samples, z_edges.shape[-1]), 2, inds_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        fine_z_vals = edges_g[..., 0] + t * (edges_g[..., 1] - edges_g[..., 0])
        
        origins = ray_bundle.origins.unsqueeze(1).expand(-1, self.n_fine_samples, -1)
        directions = ray_bundle.directions.unsqueeze(1).expand(-1, self.n_fine_samples, -1)
        fine_sample_points = origins + directions * fine_z_vals.unsqueeze(-1)
        
        return fine_sample_points, fine_z_vals

    def forward(self, ray_bundle, model, render_func=None):
        coarse_bundle, z_vals = self.sample_coarse(ray_bundle)
        
        with torch.no_grad():
            coarse_output = model(coarse_bundle)
            coarse_density = coarse_output["density"]
            
            deltas = torch.cat([
                z_vals[..., 1:] - z_vals[..., :-1],
                torch.ones_like(z_vals[..., :1]) * 1e10
            ], dim=-1).unsqueeze(-1)
            
            alpha = 1.0 - torch.exp(-coarse_density * deltas)
            weights = alpha.clone().squeeze(-1)
            T = torch.ones_like(weights)
            for i in range(1, weights.shape[-1]):
                T[..., i] = T[..., i-1] * (1.0 - alpha[..., i-1, 0] + 1e-10)
            
            weights = alpha[..., 0] * T
        
        fine_points, fine_z_vals = self.sample_fine(ray_bundle, z_vals, weights)
        
        all_points = torch.cat([coarse_bundle.sample_points, fine_points], dim=1)
        all_z_vals = torch.cat([z_vals, fine_z_vals], dim=1)
        
        _, indices = torch.sort(all_z_vals, dim=-1)
        all_z_vals = torch.gather(all_z_vals, 1, indices)
        batch_size = all_points.shape[0]
        indices = indices.unsqueeze(-1).expand(-1, -1, 3)
        all_points = torch.gather(all_points, 1, indices)
        
        final_bundle = ray_bundle._replace(
            sample_points=all_points,
            sample_lengths=all_z_vals.unsqueeze(-1)
        )
        
        return final_bundle
    
class HierarchicalNeRF(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        self.coarse_nerf = NeuralRadianceField(cfg.implicit_function)
        self.fine_nerf = NeuralRadianceField(cfg.implicit_function)
        
        self.sampler = HierarchicalSampler(cfg.sampler)
        self.renderer = None
        
    def forward_coarse(self, ray_bundle):
        coarse_bundle, z_vals = self.sampler.sample_coarse(ray_bundle)
        raw_outputs = self.coarse_nerf(coarse_bundle)
        batch_size = ray_bundle.origins.shape[0]
        n_samples = self.sampler.n_coarse_samples
        density = raw_outputs['density'].view(batch_size, n_samples, 1)
        features = raw_outputs['feature'].view(batch_size, n_samples, 3)
        deltas = torch.cat([
        z_vals[:, 1:] - z_vals[:, :-1],
        torch.ones_like(z_vals[:, :1]) * 1e10
        ], dim=-1).unsqueeze(-1)
        alpha = 1.0 - torch.exp(-density * deltas)
        weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[:, :1]),
            (1.0 - alpha + 1e-10)[:, :-1]
        ], dim=1),
        dim=1
        )
        rgb = torch.sum(weights * features, dim=1)
        
        return {
        'feature': rgb,  
        'density': density,
        'raw_outputs': raw_outputs,
        'weights': weights
        }
    
    def forward(self, ray_bundle):
        hierarchical_bundle = self.sampler(ray_bundle, self.coarse_nerf)
        raw_outputs = self.fine_nerf(hierarchical_bundle)
        batch_size = ray_bundle.origins.shape[0]
        n_samples = hierarchical_bundle.sample_points.shape[1]
        density = raw_outputs['density'].view(batch_size, n_samples, 1)
        features = raw_outputs['feature'].view(batch_size, n_samples, 3)
        z_vals = hierarchical_bundle.sample_lengths[..., 0]
        deltas = torch.cat([
        z_vals[:, 1:] - z_vals[:, :-1],
        torch.ones_like(z_vals[:, :1]) * 1e10
        ], dim=-1).unsqueeze(-1)
        alpha = 1.0 - torch.exp(-density * deltas)
        weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[:, :1]),
            (1.0 - alpha + 1e-10)[:, :-1]
        ], dim=1),
        dim=1
        )
        rgb = torch.sum(weights * features, dim=1)
        return {
        'feature': rgb, 
        'density': density,
        'weights': weights
        }

implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
    'hierarchical_sampler': HierarchicalSampler
}