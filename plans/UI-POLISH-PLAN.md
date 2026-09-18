# UI polish plan

Scope: `webgpu-demo/index.html` only. Four changes. No layout restructuring, no
palette change, no font swap, no new dependencies. The demo is being recorded
imminently, so every change must be reversible and visually verifiable.

## Design read

Reading this as: a technical inference dashboard for engineers, in an existing
Linear-adjacent dark language, leaning toward refinement of what is already
there rather than redirection.

The `taste-skill` core explicitly excludes dashboards and data tables, and
`minimalist-skill` specifies a light warm-monochrome canvas. Neither is applied
wholesale. What carries over is the aesthetic-agnostic discipline both share:
emoji are not iconography, numbers in a column must align, color is a scarce
semantic resource, and hierarchy comes from type and spacing before it comes
from decoration.

The existing token system (`--canvas` through `--radius`) is sound and stays.

---

## P1. Replace emoji with typographic and SVG marks

Current usage: 10x `⚡`, 2x `📋`, and one each of `🟢`, `🔴`, `🔀`, `✓`.

Emoji render differently per platform, sit on their own baseline, carry their own
color outside the palette, and read as placeholder iconography on a technical
tool. This is the single largest taste delta available and it is low risk.

- `⚡` in the latency badge and Run Inference button: remove entirely from the
  badge (the number carries it) and replace in the button with nothing, letting
  the label and the existing keyboard hint do the work.
- `📋` on the copy button: inline SVG, 14x14, `currentColor`, 1.5px stroke.
- `🟢` / `🔴` status dots: replace with a styled `span` using
  `--emerald` / `--rose`, 6px, `border-radius: 50%`. A `●` glyph already appears
  6 times elsewhere in the file; match that existing treatment rather than
  inventing a second one.
- `🔀` on Shuffle Option Order: inline SVG, same spec as the copy icon.
- `✓`: keep. It is a typographic glyph, not an emoji, and it renders
  consistently.

Do not introduce an icon library.

## P2. Tabular numerals on every numeric readout

The probability column (`99.5%`, `0.1%`, `0.2%`) and the latency badge change
value on every run. With proportional figures the digits jitter horizontally as
values change, which is highly visible in a screen recording.

Add `font-variant-numeric: tabular-nums;` to the probability cell, the latency
badge, the token and temperature telemetry, and the threshold value. JetBrains
Mono is already the family, so this is a one-property change per rule.

Right-align the probability column so the percent signs form a single vertical
edge.

## P3. Emphasise the winning row

Currently every candidate row carries equal weight after inference, and the
winner is distinguished only by the length of its bar. The abstention row has a
dashed amber treatment that reads stronger than the actual winner.

Give the highest-probability row, after a run completes, a single subtle
treatment: `background: var(--accent-subtle)` with the existing `--radius`, and
raise its probability figure to `--ink` while others sit at `--ink-subtle`.

One signal only. Do not add a border, a glow, a scale transform, or a shadow.
This must not fight the existing abstention styling when abstention is the
winner, in which case use `--amber-subtle` instead.

## P4. Header telemetry hierarchy

`Tokens: 103 • T: 1.00` and the timing line currently sit at similar visual
weight to the engine title. They are secondary metadata.

Drop them to `--ink-tertiary`, reduce to 11px, and keep them on one line. The
engine name and readiness state should be the only things at full `--ink` in
the header.

---

## Out of scope

Do not change: the palette, `--font-sans` (Inter stays), layout structure,
component order, preset text, any copy, `worker.js`, or any file outside
`webgpu-demo/index.html`.

Do not add animation. Motion on a screen recording of an inference tool reads as
noise, and the numbers are the content.

## Acceptance

1. No emoji remain in `webgpu-demo/index.html`.
2. Probability figures hold their horizontal position as values change between
   runs.
3. The winning row is identifiable at a glance in a still frame, including when
   the winner is abstention.
4. Header metadata is visibly subordinate to the engine title.
5. `python -m pytest tests/ -q` still passes and both render `--check` scripts
   still pass.
6. A run still produces correct probabilities, policy routing, and a downloadable
   receipt.
