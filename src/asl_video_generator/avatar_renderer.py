"""3D Avatar Renderer - Convert skeletal poses/meshes to video.

This module provides rendering functionality to convert:
- 2D skeletal poses (from SignLLM) → 2D overlay video
- 3D SMPL-H meshes (from wSignGen) → 3D avatar video

Supports multiple rendering backends:
- Matplotlib (quick preview, 2D)
- Three.js export (for web playback)
- Blender (high quality, offline)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class MeshBackendTelemetry(TypedDict):
    """Runtime telemetry for effective mesh backend usage."""

    last_backend: Literal["stylized", "software_3d", "pyrender"] | None
    counts: dict[str, int]
    pyrender_fallback_count: int


@dataclass
class RenderConfig:
    """Configuration for video rendering."""
    
    width: int = 512
    height: int = 512
    fps: int = 30
    background_color: tuple[int, int, int] = (255, 255, 255)
    avatar_style: Literal["skeleton", "mesh", "stylized"] = "skeleton"
    output_format: Literal["mp4", "webm", "gif", "frames"] = "mp4"
    # Mesh renderer backend:
    # - stylized: existing coefficient-driven 2D proxy rendering
    # - software_3d: triangle rasterization from 3D vertices/faces
    # - pyrender: reserved for future external renderer integration (falls back to software_3d)
    mesh_backend: Literal["stylized", "software_3d", "pyrender"] = "software_3d"
    # For 3D mesh rendering
    camera_distance: float = 2.5
    camera_angle: tuple[float, float] = (0.0, 0.0)  # (azimuth, elevation)


class AvatarRenderer:
    """Render ASL poses/motions to video.
    
    This class provides a unified interface for rendering both
    2D skeletal poses and 3D mesh animations to video formats.
    """
    
    def __init__(self, config: RenderConfig | None = None):
        """Initialize renderer with config."""
        self.config = config or RenderConfig()
        self._pyrender_available: bool | None = None
        self._pyrender_fallback_warned = False
        self._mesh_backend_last_used: Literal["stylized", "software_3d", "pyrender"] | None = None
        self._mesh_backend_usage_counts: dict[str, int] = {
            "stylized": 0,
            "software_3d": 0,
            "pyrender": 0,
        }
        self._pyrender_fallback_count = 0
    
    def render_poses(
        self,
        pose_path: Path,
        output_path: Path,
    ) -> Path:
        """Render 2D skeletal poses to video.
        
        Args:
            pose_path: Path to pose JSON file (from PoseGenerator)
            output_path: Path for output video
            
        Returns:
            Path to rendered video
        """
        # Load pose data
        pose_data = json.loads(pose_path.read_text())
        frames = pose_data["frames"]
        
        if self.config.output_format == "frames":
            return self._render_pose_frames(frames, output_path)
        else:
            return self._render_pose_video(frames, output_path, pose_data)
    
    def render_mesh(
        self,
        motion_path: Path,
        output_path: Path,
    ) -> Path:
        """Render 3D SMPL-H mesh animation to video.
        
        Args:
            motion_path: Path to motion JSON file (from VocabularyGenerator)
            output_path: Path for output video
            
        Returns:
            Path to rendered video
        """
        # Load motion data
        motion_data = json.loads(motion_path.read_text())
        frames = motion_data["frames"]
        
        if self.config.output_format == "frames":
            return self._render_mesh_frames(frames, output_path)
        else:
            return self._render_mesh_video(frames, output_path, motion_data)
    
    def _render_pose_frames(
        self, frames: list[dict[str, Any]], output_dir: Path
    ) -> Path:
        """Render pose frames as individual images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:05d}.png"
            self._draw_skeleton_frame(frame, frame_path)
        
        return output_dir
    
    def _render_pose_video(
        self, frames: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]
    ) -> Path:
        """Render poses as video using PIL + imageio."""
        try:
            import imageio
            import numpy as np
            from PIL import Image, ImageDraw
        except ImportError:
            print("Warning: imageio/PIL not available, saving as frames")
            return self._render_pose_frames(frames, output_path.parent / "frames")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            writer = imageio.get_writer(str(output_path), fps=self.config.fps)
            
            for i, frame in enumerate(frames):
                # Create PIL image
                img = Image.new('RGB', (self.config.width, self.config.height), 
                               self.config.background_color)
                draw = ImageDraw.Draw(img)
                
                # Draw skeleton
                scale = self.config.width
                
                # Draw body connections first (simplified)
                # Ideally, we'd draw lines between connected joints
                
                # Draw keypoints
                for point in frame.get("body", []):
                    if len(point) >= 2:
                        x, y = int(point[0] * scale), int(point[1] * scale)
                        # Ensure within bounds
                        if 0 <= x < self.config.width and 0 <= y < self.config.height:
                            draw.ellipse([x-5, y-5, x+5, y+5], fill='blue')
                
                for point in frame.get("right_hand", []):
                    if len(point) >= 2:
                        x, y = int(point[0] * scale), int(point[1] * scale)
                        if 0 <= x < self.config.width and 0 <= y < self.config.height:
                            draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
                            
                for point in frame.get("left_hand", []):
                    if len(point) >= 2:
                        x, y = int(point[0] * scale), int(point[1] * scale)
                        if 0 <= x < self.config.width and 0 <= y < self.config.height:
                            draw.ellipse([x-2, y-2, x+2, y+2], fill='green')
                
                # Add frame number/progress
                draw.text((10, 10), f"Frame {i}/{len(frames)}", fill='black')
                
                # Convert to numpy array and write
                writer.append_data(np.array(img))
                
            writer.close()
            return output_path
            
        except Exception as e:
            print(f"Error rendering video with imageio: {e}")
            # Fallback
            return self._render_pose_frames(frames, output_path.parent / "frames")
    
    def _render_mesh_frames(
        self, frames: list[dict[str, Any]], output_dir: Path
    ) -> Path:
        """Render mesh frames as individual images."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:05d}.png"
            self._draw_mesh_placeholder(frame, frame_path, i, len(frames))

        return output_dir
    
    def _render_mesh_video(
        self, frames: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]
    ) -> Path:
        """Render mesh animation as video."""
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not available, exporting mesh frames instead")
            frames_dir = output_path.parent / f"{output_path.stem}_frames"
            return self._render_mesh_frames(frames, frames_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(metadata.get("fps", self.config.fps))
        if fps <= 0:
            fps = self.config.fps

        writer = None
        try:
            if output_path.suffix.lower() == ".gif":
                writer = imageio.get_writer(str(output_path), duration=1000.0 / fps)
            else:
                writer = imageio.get_writer(str(output_path), fps=fps)
            for i, frame in enumerate(frames):
                img = self._build_mesh_image(frame, i, len(frames))
                writer.append_data(np.array(img))
            return output_path
        except Exception as e:
            print(f"Error rendering mesh video: {e}. Exporting frames instead.")
            frames_dir = output_path.parent / f"{output_path.stem}_frames"
            self._render_mesh_frames(frames, frames_dir)
            return frames_dir
        finally:
            if writer is not None:
                writer.close()

    def _sample_pose_values(self, values: list[float], count: int) -> NDArray[np.float64]:
        """Downsample or upsample pose coefficients to a fixed count."""
        if count <= 0:
            return np.array([], dtype=np.float64)

        if not values:
            return np.zeros(count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)
        if array.size == 1:
            return cast(NDArray[np.float64], np.repeat(array, count))

        source_index = np.arange(array.size, dtype=np.float64)
        target_index = np.linspace(0.0, float(array.size - 1), num=count)
        return cast(
            NDArray[np.float64],
            np.asarray(np.interp(target_index, source_index, array), dtype=np.float64),
        )

    def _build_mesh_image(
        self, frame: dict[str, Any], frame_index: int, total_frames: int
    ) -> "PILImage":
        """Dispatch mesh frame rendering to configured backend."""
        backend = self.config.mesh_backend
        if backend == "stylized":
            self._record_mesh_backend_usage("stylized")
            return self._build_mesh_image_stylized(frame, frame_index, total_frames)
        if backend == "pyrender":
            if self._is_pyrender_available():
                try:
                    img = self._build_mesh_image_pyrender(frame, frame_index, total_frames)
                    self._record_mesh_backend_usage("pyrender")
                    return img
                except Exception as exc:
                    self._record_pyrender_fallback()
                    self._warn_pyrender_fallback(
                        "pyrender backend failed "
                        f"({exc}); falling back to software_3d mesh renderer."
                    )
                    self._record_mesh_backend_usage("software_3d")
                    return self._build_mesh_image_software_3d(frame, frame_index, total_frames)
            self._record_pyrender_fallback()
            self._warn_pyrender_fallback(
                "pyrender backend unavailable; falling back to software_3d mesh renderer."
            )
            self._record_mesh_backend_usage("software_3d")
            return self._build_mesh_image_software_3d(frame, frame_index, total_frames)
        if backend == "software_3d":
            self._record_mesh_backend_usage("software_3d")
            return self._build_mesh_image_software_3d(frame, frame_index, total_frames)
        self._record_mesh_backend_usage("stylized")
        return self._build_mesh_image_stylized(frame, frame_index, total_frames)

    def _record_mesh_backend_usage(
        self, backend: Literal["stylized", "software_3d", "pyrender"]
    ) -> None:
        """Track effective mesh backend used for a rendered frame."""
        self._mesh_backend_last_used = backend
        self._mesh_backend_usage_counts[backend] += 1

    def _record_pyrender_fallback(self) -> None:
        """Track pyrender fallback events across frames."""
        self._pyrender_fallback_count += 1

    def get_mesh_backend_telemetry(self) -> MeshBackendTelemetry:
        """Return mesh backend telemetry for observability and tests."""
        return {
            "last_backend": self._mesh_backend_last_used,
            "counts": dict(self._mesh_backend_usage_counts),
            "pyrender_fallback_count": self._pyrender_fallback_count,
        }

    def _warn_pyrender_fallback(self, message: str) -> None:
        """Emit one-time warning when pyrender backend falls back to software."""
        if not self._pyrender_fallback_warned:
            print(message)
            self._pyrender_fallback_warned = True

    def _is_pyrender_available(self) -> bool:
        """Check if optional pyrender dependencies are available."""
        if self._pyrender_available is not None:
            return self._pyrender_available
        try:
            import pyrender  # noqa: F401
            import trimesh  # noqa: F401
        except Exception:
            self._pyrender_available = False
            return False
        self._pyrender_available = True
        return True

    def _build_mesh_image_pyrender(
        self, frame: dict[str, Any], frame_index: int, total_frames: int
    ) -> "PILImage":
        """Render mesh frame via optional pyrender offscreen path."""
        from PIL import Image

        vertices, faces = self._extract_or_synthesize_mesh(frame)
        rgb = self._render_with_pyrender(vertices, faces, frame)
        img = Image.fromarray(rgb, mode="RGB")
        return cast("PILImage", img)

    def _render_with_pyrender(
        self, vertices: np.ndarray, faces: np.ndarray, frame: dict[str, Any]
    ) -> NDArray[np.uint8]:
        """Render mesh using pyrender offscreen renderer."""
        import pyrender
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        bg = np.array(
            [
                self.config.background_color[0] / 255.0,
                self.config.background_color[1] / 255.0,
                self.config.background_color[2] / 255.0,
                1.0,
            ],
            dtype=np.float32,
        )
        scene = pyrender.Scene(
            bg_color=bg,
            ambient_light=np.array([0.25, 0.25, 0.25], dtype=np.float32),
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        camera = pyrender.PerspectiveCamera(yfov=float(np.pi / 3.0))
        camera_pose = self._build_pyrender_camera_pose(frame)
        scene.add(camera, pose=camera_pose)

        key_light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.2)
        fill_light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=0.9)
        scene.add(key_light, pose=camera_pose)
        fill_pose = np.array(camera_pose, copy=True)
        fill_pose[0, 3] -= 0.5
        fill_pose[1, 3] += 0.25
        scene.add(fill_light, pose=fill_pose)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.config.width,
            viewport_height=self.config.height,
        )
        try:
            color, _depth = renderer.render(scene)
        finally:
            renderer.delete()

        if color.ndim != 3:
            raise ValueError("Unexpected pyrender output shape.")

        if color.shape[2] >= 3:
            rgb = color[:, :, :3]
        else:
            raise ValueError("pyrender output missing RGB channels.")

        return np.asarray(rgb, dtype=np.uint8)

    def _build_pyrender_camera_pose(self, frame: dict[str, Any]) -> NDArray[np.float64]:
        """Construct world pose for pyrender camera from config and frame translation."""
        azimuth_deg, elevation_deg = self.config.camera_angle
        azimuth = float(np.deg2rad(azimuth_deg))
        elevation = float(np.deg2rad(elevation_deg))

        distance = max(float(self.config.camera_distance), 1.0)
        x = distance * np.sin(azimuth) * np.cos(elevation)
        y = distance * np.sin(elevation)
        z = distance * np.cos(azimuth) * np.cos(elevation)
        eye = np.array([x, y, z], dtype=np.float64)

        translation = frame.get("translation", [0.0, 0.0, 0.0])
        target: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        if isinstance(translation, (list, tuple)) and len(translation) == 3:
            target = np.array(
                [float(translation[0]), float(translation[1]), float(translation[2])],
                dtype=np.float64,
            )

        forward = target - eye
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            forward = forward / forward_norm

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / right_norm

        up = np.cross(right, forward)
        up_norm = float(np.linalg.norm(up))
        if up_norm >= 1e-6:
            up = up / up_norm

        pose: NDArray[np.float64] = np.eye(4, dtype=np.float64)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return cast(NDArray[np.float64], pose)

    def _build_mesh_image_stylized(
        self, frame: dict[str, Any], frame_index: int, total_frames: int
    ) -> "PILImage":
        """Create a stylized mesh visualization frame as a PIL image."""
        from PIL import Image, ImageDraw

        img = Image.new(
            "RGB",
            (self.config.width, self.config.height),
            self.config.background_color,
        )
        draw = ImageDraw.Draw(img)

        center_x = self.config.width // 2
        center_y = self.config.height // 2

        translation = frame.get("translation", [0.0, 0.0, 0.0])
        if isinstance(translation, (list, tuple)) and len(translation) == 3:
            center_x += int(float(translation[0]) * self.config.width * 0.45)
            center_y -= int(float(translation[2]) * self.config.height * 0.25)

        body_signal = self._sample_pose_values(frame.get("body_pose", []), 10)
        left_signal = self._sample_pose_values(frame.get("left_hand_pose", []), 15)
        right_signal = self._sample_pose_values(frame.get("right_hand_pose", []), 15)

        body_points: list[tuple[int, int]] = []
        for i, value in enumerate(body_signal):
            y = int(center_y - 80 + i * 16)
            x = int(center_x + float(value) * 45)
            body_points.append((x, y))

        if len(body_points) > 1:
            draw.line(body_points, fill=(80, 80, 80), width=4)
        for x, y in body_points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(120, 120, 120))

        left_anchor = (center_x - 36, center_y - 36)
        right_anchor = (center_x + 36, center_y - 36)

        left_points = [left_anchor]
        for i, value in enumerate(left_signal):
            angle = np.pi * (0.8 + (i / max(len(left_signal), 1)) * 0.5)
            radius = 12 + i * 3 + abs(float(value)) * 8
            x = int(left_anchor[0] + np.cos(angle) * radius)
            y = int(left_anchor[1] + np.sin(angle) * radius)
            left_points.append((x, y))

        right_points = [right_anchor]
        for i, value in enumerate(right_signal):
            angle = np.pi * (-0.3 + (i / max(len(right_signal), 1)) * 0.5)
            radius = 12 + i * 3 + abs(float(value)) * 8
            x = int(right_anchor[0] + np.cos(angle) * radius)
            y = int(right_anchor[1] + np.sin(angle) * radius)
            right_points.append((x, y))

        if len(left_points) > 1:
            draw.line(left_points, fill=(20, 140, 220), width=2)
        if len(right_points) > 1:
            draw.line(right_points, fill=(220, 90, 60), width=2)

        draw.ellipse(
            [center_x - 20, center_y - 120, center_x + 20, center_y - 80],
            outline=(90, 90, 90),
            width=3,
            fill=(235, 235, 235),
        )

        draw.text((8, 8), f"Mesh frame {frame_index + 1}/{max(total_frames, 1)}", fill=(0, 0, 0))
        return img

    def _build_mesh_image_software_3d(
        self, frame: dict[str, Any], frame_index: int, total_frames: int
    ) -> "PILImage":
        """Render mesh frame using software 3D triangle rasterization."""
        from PIL import Image, ImageDraw

        vertices, faces = self._extract_or_synthesize_mesh(frame)
        projected, depths, transformed = self._project_mesh(vertices, frame)

        face_vertices = transformed[faces]
        face_normals = np.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
        )
        normal_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.clip(normal_norm, 1e-6, None)
        light_dir = np.array([0.35, 0.4, 1.0], dtype=np.float64)
        light_dir = light_dir / np.linalg.norm(light_dir)
        face_intensity = np.clip(face_normals @ light_dir, 0.15, 1.0)

        img = Image.new(
            "RGB",
            (self.config.width, self.config.height),
            self.config.background_color,
        )
        draw = ImageDraw.Draw(img)

        # Painter's algorithm: draw far faces first.
        face_depths = depths[faces].mean(axis=1)
        face_order = np.argsort(face_depths)[::-1]

        for face_idx in face_order:
            face = faces[face_idx]
            pts = projected[face]
            shade = int(max(30.0, min(235.0, 45.0 + 170.0 * float(face_intensity[face_idx]))))
            color = (shade, int(0.95 * shade), int(0.9 * shade))
            polygon = [(int(p[0]), int(p[1])) for p in pts]
            draw.polygon(polygon, fill=color, outline=(35, 35, 35))

        draw.text(
            (8, 8),
            f"Mesh3D {frame_index + 1}/{max(total_frames, 1)}",
            fill=(0, 0, 0),
        )
        return img

    def _extract_or_synthesize_mesh(self, frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Extract vertices/faces from frame or synthesize a low-poly proxy mesh."""
        vertices_raw = frame.get("vertices")
        faces_raw = frame.get("faces")

        if isinstance(vertices_raw, list) and isinstance(faces_raw, list):
            vertices = np.asarray(vertices_raw, dtype=np.float64)
            faces = np.asarray(faces_raw, dtype=np.int64)
            valid_vertices = vertices.ndim == 2 and vertices.shape[1] == 3
            valid_faces = faces.ndim == 2 and faces.shape[1] == 3
            if valid_vertices and valid_faces:
                return vertices, faces

        body = self._sample_pose_values(frame.get("body_pose", []), 12)
        left = self._sample_pose_values(frame.get("left_hand_pose", []), 6)
        right = self._sample_pose_values(frame.get("right_hand_pose", []), 6)

        # Build a lat-long sphere-like mesh and deform it by pose coefficients.
        lat_count = 8
        lon_count = 12
        verts: list[list[float]] = []
        for i in range(lat_count):
            v = i / (lat_count - 1)
            phi = np.pi * (v - 0.5)
            for j in range(lon_count):
                u = j / lon_count
                theta = 2.0 * np.pi * u
                radius = 0.45 + 0.06 * np.sin(theta * 2.0 + body[i % max(body.size, 1)] * 2.0)
                x = radius * np.cos(phi) * np.cos(theta)
                y = radius * np.sin(phi) * 1.2
                z = radius * np.cos(phi) * np.sin(theta)
                verts.append([float(x), float(y), float(z)])

        if left.size:
            for idx in range(min(left.size, lon_count)):
                verts[idx][0] -= 0.04 * float(left[idx])
                verts[idx][2] += 0.03 * float(left[idx])
        if right.size:
            offset = lon_count
            for idx in range(min(right.size, lon_count)):
                v_idx = offset + idx
                verts[v_idx][0] += 0.04 * float(right[idx])
                verts[v_idx][2] -= 0.03 * float(right[idx])

        synthesized_faces: list[list[int]] = []
        for i in range(lat_count - 1):
            for j in range(lon_count):
                nj = (j + 1) % lon_count
                a = i * lon_count + j
                b = i * lon_count + nj
                c = (i + 1) * lon_count + j
                d = (i + 1) * lon_count + nj
                synthesized_faces.append([a, c, b])
                synthesized_faces.append([b, c, d])

        return np.asarray(verts, dtype=np.float64), np.asarray(synthesized_faces, dtype=np.int64)

    def _project_mesh(
        self, vertices: np.ndarray, frame: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D vertices into 2D screen coordinates."""
        azimuth_deg, elevation_deg = self.config.camera_angle
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)

        rot_y = np.array(
            [
                [np.cos(azimuth), 0.0, np.sin(azimuth)],
                [0.0, 1.0, 0.0],
                [-np.sin(azimuth), 0.0, np.cos(azimuth)],
            ],
            dtype=np.float64,
        )
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(elevation), -np.sin(elevation)],
                [0.0, np.sin(elevation), np.cos(elevation)],
            ],
            dtype=np.float64,
        )
        rot = rot_x @ rot_y

        transformed = vertices @ rot.T

        translation = frame.get("translation", [0.0, 0.0, 0.0])
        if isinstance(translation, (list, tuple)) and len(translation) == 3:
            transformed[:, 0] += float(translation[0])
            transformed[:, 1] += float(translation[1])
            transformed[:, 2] += float(translation[2])

        camera_z = transformed[:, 2] + max(self.config.camera_distance, 1.0)
        camera_z = np.clip(camera_z, 0.25, None)

        focal = min(self.config.width, self.config.height) * 0.9
        x_screen = self.config.width * 0.5 + focal * (transformed[:, 0] / camera_z)
        y_screen = self.config.height * 0.5 - focal * (transformed[:, 1] / camera_z)
        projected = np.stack([x_screen, y_screen], axis=1)

        return projected, camera_z, transformed
    
    def _draw_skeleton_frame(self, frame: dict[str, Any], output_path: Path) -> None:
        """Draw a single skeleton frame to image."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            print(f"Skipping frame render (PIL not available): {output_path}")
            return
        
        img = Image.new('RGB', (self.config.width, self.config.height), 
                       self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        # Draw keypoints
        scale = self.config.width
        for point in frame.get("body", []):
            if len(point) >= 2:
                x, y = int(point[0] * scale), int(point[1] * scale)
                draw.ellipse([x-5, y-5, x+5, y+5], fill='blue')
        
        for point in frame.get("right_hand", []):
            if len(point) >= 2:
                x, y = int(point[0] * scale), int(point[1] * scale)
                draw.ellipse([x-3, y-3, x+3, y+3], fill='red')
        
        img.save(output_path)
    
    def _draw_mesh_placeholder(
        self,
        frame: dict[str, Any],
        output_path: Path,
        frame_index: int = 0,
        total_frames: int = 1,
    ) -> None:
        """Draw a mesh visualization frame to image."""
        try:
            img = self._build_mesh_image(frame, frame_index, total_frames)
        except ImportError:
            return

        img.save(output_path)
    
    def export_threejs(
        self,
        motion_path: Path,
        output_path: Path,
    ) -> Path:
        """Export motion data for Three.js web playback.
        
        Creates a JSON file compatible with Three.js animation system.
        """
        motion_data = json.loads(motion_path.read_text())
        
        # Convert to Three.js AnimationClip format
        threejs_data = {
            "name": motion_data.get("word", "animation"),
            "fps": motion_data["fps"],
            "duration": motion_data["total_duration_ms"] / 1000,
            "tracks": self._convert_to_threejs_tracks(motion_data["frames"]),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(threejs_data, indent=2))
        
        return output_path
    
    def _convert_to_threejs_tracks(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert frame data to Three.js animation tracks."""
        tracks = []
        
        # Extract time values
        times = [f["timestamp_ms"] / 1000 for f in frames]
        
        # Body rotation track (simplified)
        if frames[0].get("body_pose"):
            body_values = []
            for f in frames:
                # Take first 3 values as rotation
                rot = f.get("body_pose", [0, 0, 0])[:3]
                body_values.extend(rot)
            
            tracks.append({
                "name": "Body.rotation",
                "type": "vector",
                "times": times,
                "values": body_values,
            })
        
        # Right hand track
        if frames[0].get("right_hand_pose"):
            hand_values = []
            for f in frames:
                # Flatten hand pose
                hand_values.extend(f.get("right_hand_pose", [0] * 45))
            
            tracks.append({
                "name": "RightHand.pose",
                "type": "vector",
                "times": times,
                "values": hand_values,
            })
        
        return tracks


def render_batch(
    input_dir: Path,
    output_dir: Path,
    config: RenderConfig | None = None,
) -> list[Path]:
    """Render all pose/motion files in a directory to video.
    
    Args:
        input_dir: Directory containing pose/motion JSON files
        output_dir: Directory for rendered videos
        config: Render configuration
        
    Returns:
        List of paths to rendered videos
    """
    renderer = AvatarRenderer(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for json_path in input_dir.glob("*.json"):
        if json_path.name == "manifest.json":
            continue
        
        output_path = output_dir / f"{json_path.stem}.{config.output_format if config else 'mp4'}"
        
        # Detect type based on content
        data = json.loads(json_path.read_text())
        if "body_pose" in data.get("frames", [{}])[0]:
            # SMPL-H mesh data
            result = renderer.render_mesh(json_path, output_path)
        else:
            # 2D skeletal pose data
            result = renderer.render_poses(json_path, output_path)
        
        paths.append(result)
        print(f"Rendered: {result}")
    
    return paths
