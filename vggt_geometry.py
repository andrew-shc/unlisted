from pathlib import Path
import sys
sys.path.append(str(Path("~/Documents/metrology_ir/vggt").expanduser()))

import argparse
import copy
import os
import subprocess
import tempfile

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

WNNC_SCRIPT = str(Path("~/Documents/metrology_ir/WNNC/main_wnnc.py").expanduser())


# ── Image loading ──────────────────────────────────────────────────────────────

def _tonemap_exr(path):
    """Reinhard-tonemap an EXR to a [0,1] float32 RGB numpy array."""
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read {path}")
    img = img[..., ::-1].copy().astype(np.float32)  # BGR→RGB
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    img = img / (np.exp(np.mean(np.log(lum + 1e-6))) + 1e-6)
    img = img / (1.0 + img)
    return np.clip(img, 0, 1)


def load_images_for_vggt(image_paths, hdr=False):
    """Return preprocessed [S, 3, H, W] float tensor in [0, 1]."""
    if not hdr:
        return load_and_preprocess_images([str(p) for p in image_paths], mode="pad")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for p in image_paths:
            img_np = _tonemap_exr(p)
            pil = Image.fromarray((img_np * 255).astype(np.uint8))
            tmp = Path(tmpdir) / (p.stem + ".png")
            pil.save(str(tmp))
            tmp_paths.append(str(tmp))
        return load_and_preprocess_images(tmp_paths, mode="pad")


# ── WNNC normal estimation ─────────────────────────────────────────────────────

def _run_wnnc_inline(ply_path, out_dir, width_config="l1", iters=40):
    import wn_treecode
    import trimesh

    pcd = trimesh.load(str(ply_path), process=False)
    pts = np.array(pcd.vertices, dtype=np.float32)

    bbox_scale = 1.1
    center = (pts.min(0) + pts.max(0)) / 2.0
    scale  = (pts.max(0) - pts.min(0)).max()
    pts_n  = (pts - center) * (2.0 / (scale * bbox_scale))

    pts_t  = torch.from_numpy(pts_n).contiguous().float().cuda()
    nrm_t  = torch.zeros_like(pts_t).cuda()
    b      = torch.ones(pts_t.shape[0], 1, dtype=torch.float32, device="cuda") * 0.5
    widths = torch.ones(pts_t.shape[0], dtype=torch.float32, device="cuda")

    preset = {"l0": (0.002, 0.016), "l1": (0.01, 0.04), "l2": (0.02, 0.08),
              "l3": (0.03, 0.12),   "l4": (0.04, 0.16), "l5": (0.05, 0.2)}
    wsmin, wsmax = preset[width_config]

    wn = wn_treecode.WindingNumberTreecode(pts_t)
    with torch.no_grad():
        for i in range(iters):
            ws    = wsmin + ((iters - 1 - i) / (iters - 1)) * (wsmax - wsmin)
            A_mu  = wn.forward_A(nrm_t, widths * ws)
            ATA   = wn.forward_AT(A_mu, widths * ws)
            r     = wn.forward_AT(b, widths * ws) - ATA
            A_r   = wn.forward_A(r, widths * ws)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            nrm_t = nrm_t + alpha * r
            nrm_t = F.normalize(wn.forward_G(nrm_t, widths * ws), dim=-1).contiguous() \
                    * torch.linalg.norm(nrm_t, dim=-1, keepdim=True)
        out = F.normalize(nrm_t, dim=-1)

    out_np = np.concatenate([pts, out.cpu().numpy()], axis=-1)
    out_path = Path(out_dir) / (Path(ply_path).stem + ".xyz")
    np.savetxt(str(out_path), out_np)
    return out_path


def run_wnnc(ply_path, out_dir, width_config="l1"):
    """Run WNNC; returns path to output .xyz with normals."""
    try:
        import wn_treecode  # noqa: F401 – available in metrology_ir env
        return _run_wnnc_inline(ply_path, out_dir, width_config)
    except ImportError:
        pass

    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", "metrology_ir",
         "python", WNNC_SCRIPT,
         str(ply_path), "--width_config", width_config, "--tqdm",
         "--out_dir", str(out_dir)],
        check=True,
    )
    return Path(out_dir) / (Path(ply_path).stem + ".xyz")


# ── Poisson reconstruction + mesh cleanup ──────────────────────────────────────

def poisson_and_clean(xyz_path, depth=9, density_quantile=0.05):
    """Load WNNC-normed .xyz, run Poisson, return cleaned TriangleMesh."""
    pts_nrm = np.loadtxt(str(xyz_path))
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts_nrm[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pts_nrm[:, 3:6])

    print(f"  Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # prune low-support verts (background float)
    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))

    # topological cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # keep largest connected component
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters   = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    largest        = cluster_n_tris.argmax()
    mesh.remove_triangles_by_mask(tri_clusters != largest)
    mesh.remove_unreferenced_vertices()

    return mesh


def _mesh_vertices_to_pcd(mesh, max_points=200000):
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        raise RuntimeError("Mesh has no vertices")
    if max_points is not None and len(verts) > max_points:
        idx = np.linspace(0, len(verts) - 1, num=max_points, dtype=int)
        verts = verts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    return pcd


def _load_as_point_cloud(path, sample_mesh_points=200000):
    path = Path(path)
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        return _mesh_vertices_to_pcd(mesh, max_points=sample_mesh_points)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"Could not load geometry from: {path}")
    return pcd


def _bbox_diag(pcd):
    pts = np.asarray(pcd.points)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _prescale_transform(pcd_a, pcd_b):
    pts_a = np.asarray(pcd_a.points)
    pts_b = np.asarray(pcd_b.points)
    s = _bbox_diag(pcd_b) / max(_bbox_diag(pcd_a), 1e-12)
    mu_a = pts_a.mean(0)
    mu_b = pts_b.mean(0)

    T = np.eye(4)
    T[:3, :3] = s * np.eye(3)
    T[:3, 3] = mu_b - s * mu_a
    return T, s


def _fpfh(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    )
    return down, feat


def align_mesh_similarity(mesh, ref_geom_path, voxel_fraction=0.01, sample_mesh_points=200000):
    src_pcd = _mesh_vertices_to_pcd(mesh, max_points=sample_mesh_points)
    ref_pcd = _load_as_point_cloud(ref_geom_path, sample_mesh_points=sample_mesh_points)

    diag_ref = _bbox_diag(ref_pcd)
    voxel = max(diag_ref * voxel_fraction, 1e-6)
    print(
        f"Aligning mesh -> ref | diag(src)={_bbox_diag(src_pcd):.4f}, "
        f"diag(ref)={diag_ref:.4f}, voxel={voxel:.6f}"
    )

    T_pre, s_pre = _prescale_transform(src_pcd, ref_pcd)
    src_scaled = copy.deepcopy(src_pcd)
    src_scaled.transform(T_pre)

    a_down, a_fpfh = _fpfh(src_scaled, voxel)
    b_down, b_fpfh = _fpfh(ref_pcd, voxel)
    max_corr = voxel * 10.0

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        a_down,
        b_down,
        a_fpfh,
        b_fpfh,
        mutual_filter=True,
        max_correspondence_distance=max_corr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
    )

    icp = o3d.pipelines.registration.registration_icp(
        a_down,
        b_down,
        max_correspondence_distance=voxel * 2.0,
        init=ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    T_final = icp.transformation @ T_pre
    stats = {
        "pre_scale": float(s_pre),
        "ransac_fitness": float(ransac.fitness),
        "ransac_rmse": float(ransac.inlier_rmse),
        "icp_fitness": float(icp.fitness),
        "icp_rmse": float(icp.inlier_rmse),
    }
    return T_final, stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a mesh from a Stanford-ORB scene via VGGT + WNNC + Poisson"
    )
    parser.add_argument("scene_dir", type=Path,
                        help="scene root, e.g. Stanford-ORB/blender_LDR/teapot_scene001")
    parser.add_argument("--output", type=Path, default=None,
                        help="output mesh path (default: <scene_dir>/mesh_vggt.obj)")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--hdr", action="store_true",
                        help="images are HDR EXR; applies Reinhard tonemap before VGGT")
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="world_points_conf threshold for point filtering")
    parser.add_argument("--max_images", type=int, default=None,
                        help="cap number of images used, sampled evenly across sequence (None = all)")
    parser.add_argument("--voxel_size", type=float, default=0.002,
                        help="voxel downsampling size before WNNC (0 = skip)")
    parser.add_argument("--wnnc_width_config", default="l1",
                        choices=["l0", "l1", "l2", "l3", "l4", "l5"])
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--density_quantile", type=float, default=0.05,
                        help="Poisson density percentile below which verts are pruned")
    parser.add_argument("--align_to", type=Path, default=None,
                        help="optional reference mesh/pointcloud path for similarity alignment")
    parser.add_argument("--align_voxel_fraction", type=float, default=0.01,
                        help="alignment voxel size as fraction of reference bbox diagonal")
    parser.add_argument("--align_sample_points", type=int, default=200000,
                        help="number of points sampled from mesh/reference for alignment")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    img_ext   = "*.exr" if args.hdr else "*.png"
    img_dir   = scene_dir / args.split
    msk_dir   = scene_dir / f"{args.split}_mask"

    image_paths = sorted(img_dir.glob(img_ext))
    mask_paths  = sorted(msk_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No {img_ext} images in {img_dir}")

    mask_by_stem = {p.stem: p for p in mask_paths}
    pairs = [(ip, mask_by_stem[ip.stem]) for ip in image_paths if ip.stem in mask_by_stem]
    if not pairs:
        raise FileNotFoundError(f"No matching image/mask stems between {img_dir} and {msk_dir}")
    if args.max_images and args.max_images < len(pairs):
        idx = np.linspace(0, len(pairs) - 1, num=args.max_images, dtype=int)
        pairs = [pairs[i] for i in idx]

    image_paths, mask_paths = zip(*pairs)
    print(f"Using {len(image_paths)} image/mask pairs from '{args.split}' split")

    # ── VGGT inference ──────────────────────────────────────────────────────
    print("Loading VGGT-1B...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device)

    print("Preprocessing images...")
    images_t = load_images_for_vggt(image_paths, hdr=args.hdr).to(device, dtype=dtype)

    print("Running VGGT forward pass...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images_t)

    # [S, H, W, 3] and [S, H, W]
    world_points = predictions["world_points"][0].float().cpu()
    conf         = predictions["world_points_conf"][0].float().cpu()
    rgb_images   = predictions["images"][0].float().cpu()   # [S, 3, H, W]

    _, H, W, _ = world_points.shape

    # ── Mask + confidence filtering ─────────────────────────────────────────
    print("Filtering points by mask and confidence...")
    all_pts, all_rgb = [], []

    for i, mask_path in enumerate(mask_paths):
        mask = Image.open(str(mask_path)).convert("L").resize((W, H), Image.NEAREST)
        mask_t = torch.from_numpy(np.array(mask) > 127)        # [H, W]
        valid  = mask_t & (conf[i] > args.conf_threshold)       # [H, W]

        all_pts.append(world_points[i][valid])                   # [N, 3]
        all_rgb.append(rgb_images[i].permute(1, 2, 0)[valid])   # [N, 3]

    points_np = torch.cat(all_pts).numpy()
    colors_np = np.clip(torch.cat(all_rgb).numpy(), 0, 1)
    print(f"  {len(points_np):,} points before downsampling")

    # ── Save point cloud ────────────────────────────────────────────────────
    out_dir  = (args.output.parent if args.output else scene_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "pointcloud_vggt.ply"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  {len(pcd.points):,} points after downsampling/outlier removal")

    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  Saved -> {ply_path}")

    # ── WNNC ────────────────────────────────────────────────────────────────
    print(f"Running WNNC (width_config={args.wnnc_width_config})...")
    xyz_path = run_wnnc(ply_path, out_dir, width_config=args.wnnc_width_config)
    print(f"  Normals -> {xyz_path}")

    # ── Poisson + cleanup ───────────────────────────────────────────────────
    print("Building mesh...")
    mesh = poisson_and_clean(xyz_path, depth=args.poisson_depth, density_quantile=args.density_quantile)

    # ── Optional alignment ──────────────────────────────────────────────────
    if args.align_to is not None:
        if not args.align_to.exists():
            raise FileNotFoundError(f"--align_to not found: {args.align_to}")
        T_align, st = align_mesh_similarity(
            mesh,
            args.align_to,
            voxel_fraction=args.align_voxel_fraction,
            sample_mesh_points=args.align_sample_points,
        )
        mesh.transform(T_align)
        tf_path = out_dir / "vggt_to_ref_transform.txt"
        np.savetxt(tf_path, T_align)
        print(
            f"Alignment done | pre_scale={st['pre_scale']:.6f}, "
            f"RANSAC(fit={st['ransac_fitness']:.4f}, rmse={st['ransac_rmse']:.4f}), "
            f"ICP(fit={st['icp_fitness']:.4f}, rmse={st['icp_rmse']:.4f})"
        )
        print(f"Saved transform -> {tf_path}")

    out_mesh = args.output or (scene_dir / "mesh_vggt.obj")
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print(f"Saved mesh ({len(np.asarray(mesh.vertices)):,} verts, "
          f"{len(np.asarray(mesh.triangles)):,} tris) -> {out_mesh}")


if __name__ == "__main__":
    main()
