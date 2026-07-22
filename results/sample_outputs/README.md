# Sample outputs

Curated examples from Phase 2's real (partial) MIDV-2020 sample, kept small in
number and committed to git (unlike `data/`, which is regenerated locally and
gitignored). Source document type: Latvian passport / Slovak ID (MIDV-2020,
CC BY-SA 2.5, synthetic faces via Generated Photos — see main README attribution).

| File | What it shows |
|---|---|
| `example_genuine_lva_passport_00.jpg` | Genuine base image, unmodified |
| `example_tier1_field_tamper.jpg` | Tier 1: Personal No. and expiry date digit-swapped, visible font/background-patch mismatch |
| `example_tier2_splicing_samecode.jpg` | Tier 2: photo spliced from a donor of the *same* document code — near-seamless blend |
| `example_tier2_splicing_crosscode.jpg` | Tier 2: photo spliced from a donor of a *different* document code (Latvian passport face onto a Slovak ID) — visible tone mismatch |
| `example_degraded_glare.jpg` | Degradation pipeline: simulated specular glare over the photo region |
| `example_tier4_full_synthetic.jpg` | Tier 4: fully fabricated ID (procedural layout/fields; photo is a face crop borrowed from a genuine MIDV-2020 image, itself already AI-generated) |
| `example_tier5_recapture.jpg` | Tier 5: screen-recapture simulation (moire interference + per-channel misregistration + softened contrast) on an Albanian ID |
