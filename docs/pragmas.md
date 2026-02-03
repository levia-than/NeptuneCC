# Pragmas

NeptuneCC recognizes three pragma families: kernel, halo, overlap. Pragmas are
parsed by the Clang frontend and recorded into the manifest with source offsets.

## Kernel
```cpp
#pragma neptune kernel begin tag(k0) in(x:ghosted) out(y:owned)
  { /* stencil loop nest */ }
#pragma neptune kernel end tag(k0)
```

Notes:
- `tag(...)` is required and used as the kernel symbol.
- `in(...)`/`out(...)` clauses define port order and qualifiers.
- The block must be a canonical loop nest for SCF-to-NeptuneIR matching.

## Halo
```cpp
#pragma neptune halo begin tag(h0) dm(da) field(x) kind(global_to_local_begin)
  /* user communication begin */
#pragma neptune halo end tag(h0) kind(global_to_local_end)
  /* user communication end */
```

Notes:
- Halo pragmas are semantic hints; NeptuneCC does not rewrite the calls.
- `tag(...)` is required and must match begin/end.
- Additional clauses are recorded into `manifest.json`.

## Overlap
```cpp
#pragma neptune overlap begin tag(o0) halo(h0) kernel(k0) policy(auto)
{
  #pragma neptune halo begin tag(h0) dm(da)
  begin();
  #pragma neptune kernel begin tag(k0) in(x:ghosted) out(y:owned)
  { /* stencil loop nest */ }
  #pragma neptune kernel end tag(k0)
  #pragma neptune halo end tag(h0)
  end();
}
#pragma neptune overlap end tag(o0)
```

Notes:
- Overlap blocks must contain exactly one kernel block and the matching halo
  begin/end for the referenced `halo(...)` tag.
- If the structure is ambiguous, rewrite is skipped for correctness.
