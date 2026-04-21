#!/usr/bin/env python3
"""
Compare alignment of three meshes by coloring them red, green, and blue.

Usage:
    python compare_mesh_alignment.py mesh1.ply mesh2.obj mesh3.ply [--output output.ply] [--view]

Arguments:
    mesh1, mesh2, mesh3: Path to PLY or OBJ mesh files
    --output: Save combined mesh to file (default: comparison.ply)
    --view: Open interactive viewer (requires trimesh[viewer])
"""

import argparse
import trimesh
import numpy as np
from pathlib import Path


def color_mesh(mesh, color):
    """Color a mesh with the given RGB color (0-255 or 0-1), returning colors array."""
    # Ensure color is in 0-255 range
    if isinstance(color, (list, tuple)):
        color = np.array(color, dtype=np.uint8)
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
    
    # Return mesh copy and color array for each vertex
    mesh_copy = mesh.copy()
    colors = np.tile(color, (len(mesh_copy.vertices), 1))
    return mesh_copy, colors


def compare_meshes(mesh_paths, output_path=None, view=False):
    """
    Load three meshes and color them RGB for comparison.
    
    Parameters
    ----------
    mesh_paths : list of str
        Paths to three mesh files (PLY or OBJ format)
    output_path : str, optional
        Path to save the combined mesh (default: comparison.ply)
    view : bool
        Whether to open interactive viewer (default: False)
    
    Returns
    -------
    combined : trimesh.Trimesh
        Combined mesh with all three colored meshes
    """
    if len(mesh_paths) != 3:
        raise ValueError(f"Expected 3 mesh paths, got {len(mesh_paths)}")
    
    colors = {
        0: (255, 0, 0),      # Red
        1: (0, 255, 0),      # Green
        2: (0, 0, 255),      # Blue
    }
    color_names = {0: "Red", 1: "Green", 2: "Blue"}
    
    meshes = []
    all_colors = []
    for i, mesh_path in enumerate(mesh_paths):
        print(f"Loading {color_names[i]} mesh from: {mesh_path}")
        mesh = trimesh.load(mesh_path)
        
        if isinstance(mesh, trimesh.Trimesh):
            colored_mesh, color_array = color_mesh(mesh, colors[i])
            meshes.append(colored_mesh)
            all_colors.append(color_array)
        else:
            print(f"  Warning: {mesh_path} loaded as {type(mesh)}, converting to Trimesh")
            for submesh in mesh.geometry.values():
                colored_mesh, color_array = color_mesh(submesh, colors[i])
                meshes.append(colored_mesh)
                all_colors.append(color_array)

        print(f"\nPer mesh stats [{mesh_path}]:")
        print(f"  Total vertices: {len(meshes[i].vertices)}")
        print(f"  Total faces: {len(meshes[i].faces)}")
        print(f"  Bounds: {meshes[i].bounds}")
    
    # Manually concatenate meshes while preserving vertex colors
    vertices_list = []
    faces_list = []
    colors_list = []
    vertex_offset = 0
    
    for mesh, mesh_colors in zip(meshes, all_colors):
        vertices_list.append(mesh.vertices)
        # Offset face indices by current vertex count
        faces_list.append(mesh.faces + vertex_offset)
        colors_list.append(mesh_colors)
        vertex_offset += len(mesh.vertices)
    
    # Combine all vertices and faces
    combined_vertices = np.vstack(vertices_list)
    combined_faces = np.vstack(faces_list)
    combined_colors = np.vstack(colors_list)
    
    # Create combined mesh and assign vertex colors
    combined = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False
    )
    combined.visual.vertex_colors = combined_colors
    
    print(f"\nCombined mesh stats:")
    print(f"  Total vertices: {len(combined.vertices)}")
    print(f"  Total faces: {len(combined.faces)}")
    print(f"  Bounds: {combined.bounds}")
    
    # Save if output path provided
    if output_path is None:
        output_path = "comparison.ply"
    
    combined.export(output_path)
    print(f"\nSaved to: {output_path}")
    
    # View if requested
    if view:
        print("Opening viewer...")
        combined.show()
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Compare alignment of three meshes by coloring them RGB"
    )
    parser.add_argument("meshes", nargs=3, help="Three mesh files (PLY or OBJ)")
    parser.add_argument(
        "--output", 
        default="comparison.ply",
        help="Output file path (default: comparison.ply)"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open interactive viewer"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    for mesh_path in args.meshes:
        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    
    compare_meshes(args.meshes, output_path=args.output, view=args.view)


if __name__ == "__main__":
    main()
