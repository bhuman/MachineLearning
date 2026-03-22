import numpy as np
from isaacgym import gymapi, terrain_utils, gymutil
import torch
from utils.utils import apply_randomization


class Terrain:

    def __init__(self, gym, sim, device, terrain_cfg):
        self.terrain_cfg = terrain_cfg
        self.gym = gym
        self.sim = sim
        self.device = device
        self.type = self.terrain_cfg["type"]
        self.friction_map = None

        if self.type == "plane":
            self._create_ground_plane()
        elif self.type == "trimesh":
            self._create_trimesh()
        else:
            raise ValueError(f"Invalid terrain type: {self.type}")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_cfg["static_friction"]
        plane_params.dynamic_friction = self.terrain_cfg["dynamic_friction"]
        plane_params.restitution = self.terrain_cfg["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.env_width = self.terrain_cfg["num_terrains"] * self.terrain_cfg["terrain_width"]
        self.env_length = self.terrain_cfg["terrain_length"]
        self.border_size = self.terrain_cfg["border_size"]
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.border_pixels = int(self.border_size / self.horizontal_scale)
        terrain_width_pixels = int(self.terrain_cfg["terrain_width"] / self.horizontal_scale)
        terrain_length_pixels = int(self.terrain_cfg["terrain_length"] / self.horizontal_scale)
        self.height_field_raw = np.zeros(
            (
                self.terrain_cfg["num_terrains"] * terrain_width_pixels + 2 * self.border_pixels,
                terrain_length_pixels + 2 * self.border_pixels,
            ),
            dtype=np.int16,
        )
        proportions = [
            self.terrain_cfg["num_terrains"]
            * np.sum(self.terrain_cfg["terrain_proportions"][: i + 1])
            / np.sum(self.terrain_cfg["terrain_proportions"])
            for i in range(len(self.terrain_cfg["terrain_proportions"]))
        ]
        for i in range(self.terrain_cfg["num_terrains"]):
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=terrain_width_pixels,
                length=terrain_length_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )
            if i < proportions[0]:
                pass
            elif i < proportions[1]:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=self.terrain_cfg["slope"], platform_size=3.0)
            elif i < proportions[2]:
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-0.5 * self.terrain_cfg["random_height"],
                    max_height=0.5 * self.terrain_cfg["random_height"],
                    step=0.005,
                    downsampled_scale=0.2,
                )
            elif i < proportions[3]:
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    max_height=self.terrain_cfg["discrete_height"],
                    min_size=1.0,
                    max_size=2.0,
                    num_rects=20,
                    platform_size=3.0,
                )
            else:
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=self.terrain_cfg["stairs_width"], step_height=self.terrain_cfg["step_height"], platform_size=1)
            start_x = self.border_pixels + i * terrain_width_pixels
            end_x = self.border_pixels + (i + 1) * terrain_width_pixels
            start_y = self.border_pixels
            end_y = self.border_pixels + terrain_length_pixels
            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, self.terrain_cfg["slope_threshold"]
        )

        self.height_field_torch = torch.from_numpy(self.height_field_raw).to(self.device)

        # Split terrain into patches and assign each patch a random friction
        # IssacGym has a bug, which results in always using the terrain friction instead of a combination of the terrain and the colliding shape.
        patch_size = self.terrain_cfg["patch_size"]
        verts = vertices.reshape(-1, 3)
        tris = triangles.reshape(-1, 3)

        tri_verts = verts[tris]          # (N, 3, 3)
        centroids = tri_verts.mean(axis=1)

        cx = centroids[:, 0]
        cy = centroids[:, 1]

        px = np.floor(cx / patch_size).astype(np.int32)
        py = np.floor(cy / patch_size).astype(np.int32)

        patch_ids = np.stack([px, py], axis=1)

        from collections import defaultdict

        patch_triangles = defaultdict(list)

        for tri, pid in zip(tris, patch_ids):
            patch_triangles[tuple(pid)].append(tri)

        self.friction_map = torch.zeros(len(patch_triangles), 3, dtype=torch.float, device=self.device)
        for (px, py), tris_in_patch in patch_triangles.items():
            tris_in_patch = np.array(tris_in_patch)
            unique_verts, new_indices = np.unique(tris_in_patch.flatten(), return_inverse=True)
            patch_vertices = verts[unique_verts]
            patch_triangles_local = new_indices.reshape(-1, 3)

            tm = gymapi.TriangleMeshParams()
            tm.nb_vertices = patch_vertices.shape[0]
            tm.nb_triangles = patch_triangles_local.shape[0]

            tm.transform.p.x = -self.border_size
            tm.transform.p.y = -self.border_size
            tm.transform.p.z = 0.0

            friction_val = apply_randomization(0, self.terrain_cfg["friction"])

            tm.static_friction = friction_val
            tm.dynamic_friction = friction_val

            tm.restitution = self.terrain_cfg["restitution"]

            patch_vertices = patch_vertices.astype(np.float32, copy=False)
            patch_triangles_local = patch_triangles_local.astype(np.uint32, copy=False)

            self.gym.add_triangle_mesh(
                self.sim,
                patch_vertices.flatten(order="C"),
                patch_triangles_local.flatten(order="C"),
                tm
            )

            self.friction_map[px * patch_ids[-1, 1] + py, 0] = (px + 0.5) * patch_size - self.border_size
            self.friction_map[px * patch_ids[-1, 1] + py, 1] = (py + 0.5) * patch_size - self.border_size
            self.friction_map[px * patch_ids[-1, 1] + py, 2] = friction_val

    def terrain_heights(self, base_pos):
        if self.type == "plane":
            return torch.zeros(len(base_pos), dtype=torch.float, device=self.device)
        else:
            x = self.border_pixels + base_pos[:, 0] / self.horizontal_scale
            y = self.border_pixels + base_pos[:, 1] / self.horizontal_scale
            x1 = torch.floor(x).long().clip(max=self.height_field_torch.shape[0] - 2)
            x2 = x1 + 1
            y1 = torch.floor(y).long().clip(max=self.height_field_torch.shape[1] - 2)
            y2 = y1 + 1
            heights = (
                (
                    (x2 - x) * (y2 - y) * self.height_field_torch[x1, y1]
                    + (x - x1) * (y2 - y) * self.height_field_torch[x2, y1]
                    + (x2 - x) * (y - y1) * self.height_field_torch[x1, y2]
                    + (x - x1) * (y - y1) * self.height_field_torch[x2, y2]
                )
                * self.vertical_scale
            )
            return heights.to(
                dtype=torch.float,
                device=self.device,
            )

    def draw_terrain_friction(self, env, gym, viewer):
        # Now draw
        for i in range(len(self.friction_map[:])):
            start = gymapi.Vec3(self.friction_map[i, 0], self.friction_map[i, 1], 0)
            end   = gymapi.Vec3(self.friction_map[i, 0], self.friction_map[i, 1], 10)
            color = gymapi.Vec3(self.friction_map[i, 2] / self.terrain_cfg["friction"]["range"][1], 0, 1.0 - self.friction_map[i, 2] / self.terrain_cfg["friction"]["range"][1])
            gymutil.draw_line(start, end, color, gym, viewer, env)
