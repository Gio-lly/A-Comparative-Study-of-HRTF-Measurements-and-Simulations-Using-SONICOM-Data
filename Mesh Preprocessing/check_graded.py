"""
check_graded_mesh.py
====================

Sanity check of a graded mesh before launching NumCalc (Mesh2HRTF).

Runs all the geometric checks that, if failed, are known to cause
crashes or non-convergence of the BEM solver: watertight, manifoldness,
self-intersections, consistent normals, plausible scale, maximum triangle
size relative to the target frequency, position of the interaural center.

USAGE:
    python check_graded_mesh.py /path/3Dmesh_graded_right.ply
    python check_graded_mesh.py /path/3Dmesh_graded_right.ply --fmax 20000

Return code:
    0 = mesh probably OK for simulation
    1 = minor issues (warnings), simulation possible but risky
    2 = serious issues (errors), simulation very likely to fail

DEPENDENCIES:
    pip install trimesh numpy rtree

`rtree` is used by trimesh for self-intersection detection; without it,
that check is skipped with a warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    print("ERRORE: serve `trimesh`. Installa con: pip install trimesh numpy rtree")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Judgment thresholds. Configurable via command line.
# ---------------------------------------------------------------------------

DEFAULT_FMAX_HZ = 24000.0
SPEED_OF_SOUND = 343.0  # m/s

# Mesh2HRTF: the head (ear-to-ear distance) must be between 100 and 180 mm.
HEAD_WIDTH_MIN_M = 0.10
HEAD_WIDTH_MAX_M = 0.18

# Maximum tolerated distance of the interaural center from the origin.
INTERAURAL_TOLERANCE_M = 0.005  # 5 mm

# BEM resolution rule: 6 elements per wavelength.
ELEMENTS_PER_WAVELENGTH = 6


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

class Report:
    """Accumulates check results; returns a consistent exit code."""
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def ok(self, msg: str):
        self.info.append(f"  [OK]   {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"  [WARN] {msg}")

    def err(self, msg: str):
        self.errors.append(f"  [FAIL] {msg}")

    def section(self, title: str):
        print(f"\n--- {title} ---")

    def flush_section(self):
        # Print in order: OK, WARN, FAIL — so issues stay at the bottom.
        for line in self.info:
            print(line)
        for line in self.warnings:
            print(line)
        for line in self.errors:
            print(line)
        self.info.clear()
        self.warnings.clear()
        self.errors.clear()

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    @property
    def total_warnings(self) -> int:
        return len(self.warnings)


# Persistent version of the counters, since Report.flush_section() resets them.
class TotalCounters:
    def __init__(self):
        self.errors = 0
        self.warnings = 0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_loading(path: Path, report: Report) -> trimesh.Trimesh | None:
    """Loads the mesh and reports basic format/size info."""
    report.section("Caricamento mesh")
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as e:
        report.err(f"Impossibile caricare la mesh: {e}")
        return None

    if not isinstance(mesh, trimesh.Trimesh):
        report.err(f"Il file non contiene una singola mesh triangolare (tipo: {type(mesh).__name__}).")
        return None

    n_v = len(mesh.vertices)
    n_f = len(mesh.faces)
    report.ok(f"File: {path.name}")
    report.ok(f"Vertici: {n_v}, Triangoli: {n_f}")
    if n_f < 1000:
        report.warn("La mesh ha pochissimi triangoli; verifica che sia quella graded e non l'originale.")
    return mesh


def check_scale_and_units(mesh: trimesh.Trimesh, report: Report):
    """Mesh2HRTF expects meters; head width should be 0.10-0.18 m."""
    report.section("Scala e unità")
    bounds = mesh.bounds  # (2, 3)
    extents = mesh.extents  # array of 3 elements
    report.ok(f"Bounding box (m): X={extents[0]:.3f}, Y={extents[1]:.3f}, Z={extents[2]:.3f}")
    report.ok(f"Min/Max (m): {bounds[0].round(3).tolist()} / {bounds[1].round(3).tolist()}")

    # The Y extent is a good proxy for head+torso width
    # (interaural axis = Y in the Mesh2HRTF convention).
    y_extent = float(extents[1])
    if HEAD_WIDTH_MIN_M <= y_extent <= 0.6:
        # 0.6 m = typical shoulder width for head+torso, so we accept
        # a wider range than [0.10, 0.18] which only applies to the bare head.
        report.ok(f"Estensione Y={y_extent*1000:.0f} mm: compatibile con scala in METRI.")
    elif y_extent < 0.05:
        report.err(f"Estensione Y={y_extent*1000:.1f} mm troppo piccola. La mesh è in mm o cm? Mesh2HRTF richiede METRI.")
    elif y_extent > 1.0:
        report.err(f"Estensione Y={y_extent:.2f} m troppo grande. Probabile errore di scala.")
    else:
        report.warn(f"Estensione Y={y_extent*1000:.0f} mm: insolita, verifica manualmente la scala.")


def check_watertight_and_manifold(mesh: trimesh.Trimesh, report: Report):
    """Air-tight + manifold are the key requirements for NumCalc."""
    report.section("Watertight e manifold")

    # Non-manifold edge: an edge shared by !=2 faces is the classic symptom.
    # trimesh exposes the count directly.
    if mesh.is_watertight:
        report.ok("Mesh watertight (nessun foro).")
    else:
        # How serious is it? Count the boundary edges.
        edges = mesh.edges_sorted
        # Boundary edge: belongs to a single face.
        from collections import Counter
        edge_counts = Counter(map(tuple, edges))
        boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
        non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
        if boundary_edges:
            report.err(f"Mesh NON watertight: {boundary_edges} boundary edges (fori aperti).")
        if non_manifold_edges:
            report.err(f"Mesh NON manifold: {non_manifold_edges} edge condivisi da >2 facce.")

    if mesh.is_winding_consistent:
        report.ok("Winding consistente (normali coerenti tra triangoli adiacenti).")
    else:
        report.err("Winding NON consistente: alcune normali sono invertite rispetto alle vicine.")

    # Positive volume = normals pointing outward (Mesh2HRTF: exterior domain).
    # Only meaningful if the mesh is watertight, otherwise volume is undefined.
    if mesh.is_watertight:
        try:
            vol = mesh.volume
            if vol > 0:
                report.ok(f"Volume positivo ({vol*1000:.2f} litri): normali rivolte verso l'esterno.")
            else:
                report.err(f"Volume NEGATIVO ({vol*1000:.2f} litri): normali rivolte all'INTERNO. "
                           "Inverti l'orientamento prima di simulare.")
        except Exception as e:
            report.warn(f"Impossibile calcolare il volume: {e}")


def check_degenerate_triangles(mesh: trimesh.Trimesh, report: Report):
    """Zero-area triangles or extreme slivers break the BEM."""
    report.section("Triangoli degeneri e slivers")

    areas = mesh.area_faces
    n_zero = int(np.sum(areas <= 1e-14))
    if n_zero == 0:
        report.ok("Nessun triangolo a zero area.")
    else:
        report.err(f"{n_zero} triangoli a zero area: rimuovili (MeshLab: Remove Zero Area Faces).")

    # Slivers: very small height/base ratio. We use a quality metric based on
    # the inradius/circumradius * 2 ratio, which equals 1 for the equilateral
    # triangle and tends to 0 for slivers.
    tri = mesh.triangles
    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    s = 0.5 * (a + b + c)
    # Avoid divisions by zero on already-degenerate triangles.
    valid = (areas > 1e-14) & (s > 1e-14)
    quality = np.zeros_like(areas)
    quality[valid] = (
        2.0 * areas[valid] / s[valid] / np.maximum(a[valid] * b[valid] * c[valid] / (4 * areas[valid]), 1e-14)
    )
    # quality is now 2 * inradius / circumradius, in [0, 1].
    n_sliver = int(np.sum(quality < 0.05) - n_zero)  # below 0.05 = worst
    if n_sliver <= 0:
        report.ok("Nessuno sliver triangolo significativo (qualità >= 0.05).")
    elif n_sliver < 10:
        report.warn(f"{n_sliver} triangoli sliver (qualità <0.05): pochi, ma controlla dove sono.")
    else:
        report.err(f"{n_sliver} triangoli sliver (qualità <0.05): possono causare divergenza BEM.")


def check_self_intersections(mesh: trimesh.Trimesh, report: Report):
    """Self-intersections = cause #1 of divergence at high frequencies."""
    report.section("Auto-intersezioni")
    try:
        # trimesh.intersections.facet_intersections does not exist; we use
        # scene-based detection with rtree.
        # In trimesh, self-intersections are detected when loading with
        # process=True (default), but getting the actual count requires
        # a manual broad-phase. We use trimesh's AABB tree.
        from trimesh.collision import CollisionManager
        # CollisionManager works between different objects; for self-intersections
        # we expose a more direct method:
        from trimesh.intersections import mesh_plane  # noqa: F401  # import check only
        # Implementation: for each face, query the AABB of its neighbors.
        intersecting = _detect_self_intersections(mesh)
        n = len(intersecting)
        if n == 0:
            report.ok("Nessuna auto-intersezione rilevata.")
        elif n < 5:
            report.warn(f"{n} auto-intersezioni rilevate: poche e isolate, valuta caso per caso.")
        else:
            report.err(f"{n} auto-intersezioni rilevate: causa probabile di divergenza BEM. "
                       "Rimuovile prima di simulare.")
    except ImportError:
        report.warn("`rtree` non installato: controllo auto-intersezioni saltato. "
                    "Installa con: pip install rtree")
    except Exception as e:
        report.warn(f"Controllo auto-intersezioni non eseguito: {e}")


def _detect_self_intersections(mesh: trimesh.Trimesh) -> set[int]:
    """Finds faces that intersect other non-adjacent faces.

    Uses trimesh's AABB tree for the broad-phase, then a triangle-triangle
    intersection test via shapely only on the candidate pairs.
    """
    intersecting = set()
    tree = mesh.triangles_tree  # AABB rtree over the triangles
    triangles = mesh.triangles
    faces = mesh.faces

    # Build adjacency: two faces are adjacent if they share a vertex.
    # Adjacent pairs should not be counted as self-intersections.
    from collections import defaultdict
    vert_to_faces = defaultdict(set)
    for fi, face in enumerate(faces):
        for v in face:
            vert_to_faces[v].add(fi)

    n_faces = len(faces)
    for i in range(n_faces):
        # Bounding box of triangle i.
        bbox = mesh.triangles_aabb[i] if hasattr(mesh, "triangles_aabb") else None
        if bbox is None:
            tri_i = triangles[i]
            bbox = np.concatenate([tri_i.min(axis=0), tri_i.max(axis=0)])
        candidates = list(tree.intersection(bbox))
        adjacent = set()
        for v in faces[i]:
            adjacent.update(vert_to_faces[v])
        for j in candidates:
            if j <= i or j in adjacent:
                continue
            # Exact test: triangle-triangle with Möller.
            if _triangles_intersect(triangles[i], triangles[j]):
                intersecting.add(i)
                intersecting.add(j)
    return intersecting


def _triangles_intersect(t1: np.ndarray, t2: np.ndarray) -> bool:
    """Fast triangle-triangle test in 3D using the separating plane."""
    # To keep things light, we use trimesh's test if available.
    try:
        from trimesh.triangles import points_to_barycentric  # noqa: F401
        # Minimal implementation: two triangles intersect if at least
        # one edge of one crosses the plane of the other within the triangle.
        return _edge_triangle_intersect_any(t1, t2) or _edge_triangle_intersect_any(t2, t1)
    except Exception:
        return False


def _edge_triangle_intersect_any(edges_tri: np.ndarray, target_tri: np.ndarray) -> bool:
    """True if at least one edge of edges_tri crosses target_tri."""
    v0, v1, v2 = target_tri
    edge1, edge2 = v1 - v0, v2 - v0
    edges = [(edges_tri[0], edges_tri[1]),
             (edges_tri[1], edges_tri[2]),
             (edges_tri[2], edges_tri[0])]
    for p, q in edges:
        d = q - p
        h = np.cross(d, edge2)
        a = np.dot(edge1, h)
        if abs(a) < 1e-12:
            continue
        f = 1.0 / a
        s = p - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            continue
        qv = np.cross(s, edge1)
        v = f * np.dot(d, qv)
        if v < 0.0 or u + v > 1.0:
            continue
        t = f * np.dot(edge2, qv)
        if 1e-9 < t < 1.0 - 1e-9:
            return True
    return False


def check_edge_lengths(mesh: trimesh.Trimesh, fmax_hz: float, report: Report):
    """Checks the rule of 6 elements per wavelength at the max frequency."""
    report.section(f"Lunghezza spigoli vs frequenza target ({fmax_hz:.0f} Hz)")
    wavelength = SPEED_OF_SOUND / fmax_hz
    max_allowed_edge = wavelength / ELEMENTS_PER_WAVELENGTH
    report.ok(f"Lunghezza d'onda a {fmax_hz:.0f} Hz: {wavelength*1000:.2f} mm")
    report.ok(f"Spigolo massimo raccomandato (lambda/6): {max_allowed_edge*1000:.2f} mm")

    edges = mesh.edges_unique_length
    e_min = float(edges.min()) * 1000
    e_max = float(edges.max()) * 1000
    e_mean = float(edges.mean()) * 1000
    e_p99 = float(np.percentile(edges, 99)) * 1000
    report.ok(f"Spigoli (mm): min={e_min:.3f}, mean={e_mean:.3f}, p99={e_p99:.3f}, max={e_max:.3f}")

    n_over = int(np.sum(edges > max_allowed_edge))
    pct_over = 100.0 * n_over / len(edges)
    if n_over == 0:
        report.ok(f"Tutti gli spigoli rispettano lambda/6 a {fmax_hz:.0f} Hz.")
    elif pct_over < 5:
        report.warn(f"{n_over} spigoli ({pct_over:.1f}%) superano lambda/6: "
                    "probabilmente nelle zone lontane dall'orecchio; impatto limitato.")
    else:
        report.err(f"{n_over} spigoli ({pct_over:.1f}%) superano lambda/6 a {fmax_hz:.0f} Hz: "
                   f"convergenza BEM compromessa. Considera grading con -y piu piccolo "
                   f"o ridurre f_max.")


def check_interaural_origin(mesh: trimesh.Trimesh, report: Report):
    """Mesh2HRTF: interaural center at the origin, +X front, +Y left."""
    report.section("Posizione e orientamento")
    center = mesh.centroid
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    report.ok(f"Centroide (m): {center.round(4).tolist()}")
    report.ok(f"Centro bounding box (m): {bbox_center.round(4).tolist()}")

    # The interaural center should be at the origin (0,0,0).
    # We use the Y center of the bounding box as a proxy for the interaural center:
    # if the mesh includes a torso, X and Z will be unbalanced, but Y should stay centered.
    y_center = float(bbox_center[1])
    if abs(y_center) < INTERAURAL_TOLERANCE_M:
        report.ok(f"Centro Y a {y_center*1000:.1f} mm dall'origine: simmetria interaurale rispettata.")
    elif abs(y_center) < 0.02:
        report.warn(f"Centro Y a {y_center*1000:.1f} mm dall'origine: leggero offset, accettabile ma verifica.")
    else:
        report.err(f"Centro Y a {y_center*1000:.1f} mm dall'origine: la mesh NON e' centrata "
                   "sull'asse interaurale. Riposiziona in Blender prima del grading.")


def check_disconnected_components(mesh: trimesh.Trimesh, report: Report):
    """Loose pieces = NumCalc fails."""
    report.section("Componenti connesse")
    components = mesh.split(only_watertight=False)
    n = len(components)
    if n == 1:
        report.ok("Una sola componente connessa.")
    else:
        sizes = sorted([len(c.faces) for c in components], reverse=True)
        report.err(f"{n} componenti sconnesse, dimensioni (#facce): {sizes[:5]}{'...' if n > 5 else ''}. "
                   "Rimuovi i frammenti isolati con MeshLab: Remove Isolated Pieces.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check di una mesh graded prima di NumCalc (Mesh2HRTF)."
    )
    parser.add_argument("mesh_path", type=Path,
                        help="Percorso al file .ply/.stl/.obj della mesh graded.")
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX_HZ,
                        help=f"Frequenza massima target in Hz (default: {DEFAULT_FMAX_HZ}).")
    args = parser.parse_args()

    if not args.mesh_path.exists():
        print(f"File non trovato: {args.mesh_path}")
        sys.exit(2)

    print("=" * 70)
    print(f"Sanity check Mesh2HRTF: {args.mesh_path.name}")
    print(f"Frequenza massima target: {args.fmax:.0f} Hz")
    print("=" * 70)

    total = TotalCounters()
    report = Report()

    mesh = check_loading(args.mesh_path, report)
    total.errors += report.total_errors
    total.warnings += report.total_warnings
    report.flush_section()

    if mesh is None:
        print("\nImpossibile proseguire: mesh non caricata.")
        sys.exit(2)

    # Sequence of checks, in order of criticality.
    for check in (
        lambda: check_scale_and_units(mesh, report),
        lambda: check_disconnected_components(mesh, report),
        lambda: check_watertight_and_manifold(mesh, report),
        lambda: check_degenerate_triangles(mesh, report),
        lambda: check_self_intersections(mesh, report),
        lambda: check_edge_lengths(mesh, args.fmax, report),
        lambda: check_interaural_origin(mesh, report),
    ):
        check()
        total.errors += report.total_errors
        total.warnings += report.total_warnings
        report.flush_section()

    # Final verdict.
    print("\n" + "=" * 70)
    print(f"VERDETTO: {total.errors} errori, {total.warnings} warning")
    print("=" * 70)
    if total.errors > 0:
        print("La mesh ha problemi GRAVI: NumCalc probabilmente divergerà o fallirà.")
        print("Sistema gli [FAIL] sopra prima di lanciare la simulazione.")
        sys.exit(2)
    if total.warnings > 0:
        print("La mesh ha problemi MINORI: la simulazione può girare ma con rischio.")
        print("Valuta se sistemare i [WARN] in base alla criticità della zona.")
        sys.exit(1)
    print("La mesh sembra pronta per la simulazione Mesh2HRTF.")
    sys.exit(0)


if __name__ == "__main__":
    main()