# Licensing

**drawing-takeoff is licensed under the GNU Affero General Public License,
version 3 or later (AGPL-3.0-or-later).** The full text is in [`LICENSE`](LICENSE).

    Copyright (C) 2026 Abraham Borg

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
    General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

This document explains *why* that license, what it does and does not restrict,
and how to grant commercial exceptions.

---

## Why AGPL is forced here

`geometry.py` imports **PyMuPDF**, which is distributed under
**AGPL-3.0-or-later or a paid Artifex commercial license** — there is no
permissive option. Anything distributed that links PyMuPDF is a combined work,
so it must be offered under AGPL-3.0-or-later too. Every other dependency is
permissive (see the inventory below), so PyMuPDF alone sets the floor.

This is not a free choice until the PyMuPDF dependency is gone.

## What the AGPL actually restricts

AGPL-3.0 is the most restrictive license the Open Source Initiative approves.
Anyone who forks this code may use and modify it, but:

- **Distribute a binary or a fork → publish the complete corresponding source**,
  under AGPL, to whoever received it.
- **Run a modified version as a network service → publish the source to every
  user of that service** (§13, the "remote network interaction" clause). This
  is the clause plain GPL lacks, and it is what stops a competitor from taking
  the takeoff engine, improving it privately, and selling it as closed SaaS.
- **No relicensing.** A fork cannot be made proprietary, and no one may add
  further restrictions on top (§7).

## What the AGPL does *not* restrict

**It does not prohibit commercial use.** No OSI-approved open source license
can: clause 6 of the Open Source Definition forbids discriminating against any
field of endeavor, commerce included. A company may sell services built on this
code — it simply cannot keep its modifications secret.

If a strict no-commercial-use rule matters more than the "open source" label,
see [Option B](#option-b-drop-pymupdf-then-go-non-commercial) below.

## Option A: dual licensing (recommended)

"Free to fork, but not commercial without my permission" is achievable today,
as a **dual license**. As the sole copyright holder you can offer the same code
under two terms at once:

1. **AGPL-3.0-or-later** — the public default in `LICENSE`. Free to use, fork,
   and study; copyleft applies.
2. **A separate commercial license** — sold or granted case by case to anyone
   who wants to build on this code *without* the obligation to publish their
   source.

Copyleft is the pressure and the commercial license is the release valve. Users
who are fine publishing source pay nothing; users who need a proprietary
product negotiate terms. This is the same model Artifex runs on PyMuPDF itself.

### Two conditions before selling a commercial license

- **You must own all the copyright.** A dual license only works if every line
  is yours to relicense. Before accepting outside contributions, require a
  Contributor License Agreement (or copyright assignment) — otherwise a
  contributor's AGPL-only patch permanently blocks proprietary relicensing of
  that file. Also confirm you hold, or have been assigned, copyright in any
  code written under employment or contract.
- **You cannot sublicense PyMuPDF.** Your commercial license covers *your*
  code only. A commercial licensee still needs their own Artifex commercial
  license for PyMuPDF, or must run the code with a permissive backend
  (see below). Say this explicitly in any commercial terms you offer.

## Option B: drop PyMuPDF, then go non-commercial

A true non-commercial license — **PolyForm Noncommercial 1.0.0** is the
best-drafted one; CC BY-NC and the Business Source License are the other
common choices — **cannot be applied while PyMuPDF is a dependency.** AGPL §7
forbids adding further restrictions to a combined work, and a
non-commercial field-of-use limit is exactly such a restriction.

Note that these are **source-available**, not open source: they fail OSD #6, so
the project would no longer be open source in the OSI sense, would be ineligible
for some package ecosystems and corporate approval paths, and would lose the
contributor goodwill that the label carries.

If that trade is still worth it, the dependency swap comes first. The
architecture already anticipates it — `geometry.py` is the **only** module that
imports `fitz`, and `tests/test_diagnose.py` already exercises the
import-fails path. Permissive replacements:

| Backend | License | Notes |
| --- | --- | --- |
| `pypdfium2` | BSD-3-Clause / Apache-2.0 | PDFium bindings; rendering and page objects |
| `pdfplumber` / `pdfminer.six` | MIT | exposes lines, curves, rects and words — closest to `get_drawings()` / `get_text("words")` |
| `pikepdf` | MPL-2.0 | file-level copyleft only; fine to combine, no reciprocal effect on your code |

The surface to replace is non-trivial: `get_drawings()`, `get_text("words")`,
`get_pixmap()` rendering with matrix/clip, `new_shape()` / `draw_polyline()` /
`insert_text()` markup, and `Pixmap` conversion. Budget it as real work, not a
find-and-replace.

## Dependency license inventory

Verified against PyPI metadata.

| Package | License | Copyleft? |
| --- | --- | --- |
| **PyMuPDF** | **AGPL-3.0-or-later** or Artifex commercial | **yes — strong, network** |
| anthropic | MIT | no |
| tiktoken | MIT | no |
| openpyxl | MIT | no |
| platformdirs | MIT | no |
| customtkinter | MIT | no |
| tkinterdnd2 | MIT | no |
| httpx | BSD-3-Clause | no |
| pydantic | MIT | no |
| requests | Apache-2.0 | no |
| certifi | MPL-2.0 | file-level only — no effect on this project |

PyMuPDF is the only dependency that constrains the outbound license.

## Keeping the boundary clean

Keep every PyMuPDF call inside `geometry.py`. It keeps the backend swappable,
which is what makes Option B reachable and what lets a commercial licensee
substitute a permissive backend without touching the rest of the engine.
