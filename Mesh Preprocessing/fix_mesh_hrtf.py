#!/usr/bin/env python3
"""
Diagnosis and repair of a head/torso mesh before hrtf_mesh_grading.

Usage:
    python fix_mesh_hrtf.py  input.stl                      # DIAGNOSIS only
    python fix_mesh_hrtf.py  input.stl  output.ply          # diagnosis + SOFT repair
    python fix_mesh_hrtf.py  input.stl  output.ply --poisson  # POISSON reconstruction

Requires:  pip install trimesh pymeshlab numpy
"""
import sys


def diagnose(path):
    import trimesh, numpy as np
    print(f"\n===== DIAGNOSI: {path} =====")
    # process=True welds coincident vertices: essential, because the STL
    # is 'triangle soup' (each triangle has its own vertices, no connectivity)
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))

    print(f"vertici (dopo saldatura): {len(m.vertices)}   facce: {len(m.faces)}")
    ext = m.extents
    print(f"bounding box: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}  "
          f"(larghezza testa attesa ~100-180 mm)")
    print(f"watertight (stagna):   {m.is_watertight}")
    print(f"winding coerente:      {m.is_winding_consistent}")
    print(f"gusci separati:        {m.body_count}   (deve essere 1)")

    # open boundaries (holes) = edges used by only one face
    edges = m.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    print(f"edge di bordo (buchi): {open_edges}   (deve essere 0)")
    print(f"edge non-manifold:     {nonmanifold}   (deve essere 0)")

    ok = (m.is_watertight and m.body_count == 1 and nonmanifold == 0)
    if ok:
        print("\n  OK: mesh stagna e manifold -> dovrebbe gradare senza crash.")
    else:
        print("\n  ! DIFETTI PRESENTI -> e' la causa del Segmentation fault.")
        if not m.is_watertight or open_edges:
            print("    - ci sono buchi/bordi aperti da chiudere")
        if m.body_count > 1:
            print("    - ci sono frammenti staccati da eliminare")
        if nonmanifold:
            print("    - ci sono edge non-manifold da riparare")
    print("=" * 42)


def _meshset(in_path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    return ms, pymeshlab


def repair_soft(in_path, out_path):
    print(f"\n===== RIPARAZIONE SOFT -> {out_path} =====")
    ms, pymeshlab = _meshset(in_path)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    # keep only the largest shell (removes fragments)
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.PercentageValue(20))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=400)
    ms.meshing_re_orient_faces_coherently()
    ms.save_current_mesh(out_path)
    print(f"  salvato: {out_path}")


def repair_poisson(in_path, out_path, depth=11):
    print(f"\n===== POISSON (guscio chiuso garantito) -> {out_path} =====")
    ms, pymeshlab = _meshset(in_path)
    ms.meshing_remove_duplicate_vertices()
    ms.compute_normal_for_point_clouds()  # Poisson needs the normals
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
    ms.meshing_re_orient_faces_coherently()
    ms.save_current_mesh(out_path)
    print(f"  salvato: {out_path} (depth={depth})  "
          f"-- Poisson arrotonda la conca: ricontrolla le orecchie")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__); sys.exit(1)
    diagnose(args[0])
    if len(args) >= 2:
        if "--poisson" in sys.argv:
            repair_poisson(args[0], args[1])
        else:
            repair_soft(args[0], args[1])
        diagnose(args[1])
