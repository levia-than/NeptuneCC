# Overlap (halo begin/end)

Overlap rewriting splits a stencil into safe-interior and boundary faces so
communication can overlap with compute.

## Region definitions
Given `bounds.lb`, `bounds.ub`, and `radius` from `neptune.ir.apply`:
- **Safe-interior**: `[lb + r, ub - r)`
- **Boundary faces** (rank<=3 only): fixed slabs that exclude interior
  (2D: top/bottom/left/right, 3D: z/y/x faces).

If the safe-interior is empty or rank>3, overlap rewrite is skipped.

## Rewrite order
Within an overlap block, the rewritten order is:
1) halo begin statements (original code)
2) `k*_interior(...)`
3) halo end statements (original code)
4) `k*_face_*` calls

This preserves correctness while enabling communication/compute overlap.
