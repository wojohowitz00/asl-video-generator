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
from typing import Literal

import numpy as np


@dataclass
class RenderConfig:
    """Configuration for video rendering."""
    
    width: int = 512
    height: int = 512
    fps: int = 30
    background_color: tuple[int, int, int] = (255, 255, 255)
    avatar_style: Literal["skeleton", "mesh", "stylized"] = "skeleton"
    output_format: Literal["mp4", "webm", "gif", "frames"] = "mp4"
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
        self, frames: list[dict], output_dir: Path
    ) -> Path:
        """Render pose frames as individual images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:05d}.png"
            self._draw_skeleton_frame(frame, frame_path)
        
        return output_dir
    
    def _render_pose_video(
        self, frames: list[dict], output_path: Path, metadata: dict
    ) -> Path:
        """Render poses as video using PIL + imageio."""
        try:
            import imageio
            from PIL import Image, ImageDraw
            import numpy as np
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
        self, frames: list[dict], output_dir: Path
    ) -> Path:
        """Render mesh frames as individual images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement 3D mesh rendering with PyRender or Blender
        # For now, create placeholder frames
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:05d}.png"
            self._draw_mesh_placeholder(frame, frame_path)
        
        return output_dir
    
    def _render_mesh_video(
        self, frames: list[dict], output_path: Path, metadata: dict
    ) -> Path:
        """Render mesh animation as video."""
        # TODO: Implement actual mesh rendering
        # For now, render as frames and combine
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        self._render_mesh_frames(frames, frames_dir)
        
        # TODO: Use ffmpeg to combine frames
        print(f"TODO: Combine frames from {frames_dir} to {output_path}")
        return frames_dir
    
    def _draw_skeleton_frame(self, frame: dict, output_path: Path) -> None:
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
    
    def _draw_mesh_placeholder(self, frame: dict, output_path: Path) -> None:
        """Draw placeholder for mesh frame."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        
        img = Image.new('RGB', (self.config.width, self.config.height),
                       self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        # Draw simple avatar silhouette
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        
        # Head
        draw.ellipse([center_x-30, center_y-100, center_x+30, center_y-40], 
                    fill='gray')
        # Body
        draw.rectangle([center_x-40, center_y-40, center_x+40, center_y+60],
                      fill='gray')
        
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
    
    def _convert_to_threejs_tracks(self, frames: list[dict]) -> list[dict]:
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
