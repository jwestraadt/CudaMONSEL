# Why is the observed dark-γ′ BSE contrast so strong? — four studies

René 65 γ/γ′. The single-probe/carbon decks gave only ~1–3% BSE contrast, yet
strong dark-γ′ is observed and **grows with repeated scanning**. These four
studies (each in its own folder; decks are committed, outputs are git-ignored)
isolate the candidate mechanisms.

Run a deck with `..\..\x64\Release\CudaMONSEL.exe studies\<n>_*\<deck>.json` from
the project dir, then `python tools\study_analyze.py [energy|carbon|oxide|filter]`.

Background number: the intrinsic **Heinrich (mass-averaged) BSE Z-contrast is
−5.8%** dark-γ′, almost entirely from **W/Mo depletion in γ′** (γ carries ~13%
W+Mo by mass, γ′ ~3%).

## Exp 1 — energy sweep (bulk γ vs γ′, `bulk_yield`)
`energy_sweep_contrast.png`. The intrinsic BSE Z-contrast is **~0 at 200 eV–1 keV**
and only grows to **−4 to −6%** (toward the Heinrich limit) at ≥3 keV. SE contrast
is ~+18% bright-γ′ (lower φ) at all energies; the **total inverts bright→dark
around ~4 keV**. → At the low *landing* energies used with beam deceleration the
intrinsic BSE contrast is genuinely weak; the strong dark-γ′ is **not** intrinsic
low-energy BSE.

## Exp 2 — thicker uniform carbon (cumulative-scanning proxy, 1 keV)
`diff_carbon_contrast.png`. Carbon strips the bright-γ′ SE and the total inverts,
but it **saturates at the BSE floor (~−2.5%) by ~1 nm** and does **not** deepen out
to 5 nm. → A *uniform* contamination layer, however thick, cannot by itself
produce strong dark-γ′.

## Exp 3 — γ′ oxide skin ((Al,Ti)-oxide precipitate, bounding case)
`gp_oxide_images.png`. A light Al/Ti-oxide gives **BSE −24% (200 eV) and −44%
(1 keV)**; at 1 keV the precipitate goes jet-black (total −20%). γ′ is the
**Al/Ti-rich** phase (~10% Al + 10% Ti vs ~1% in γ), so it preferentially forms a
low-Z oxide/reaction skin under the beam. → A **γ′-selective light surface layer**
is a powerful dark-γ′ mechanism, and it is cumulative with exposure — the best
match to the scanning-enhanced observation.

## Exp 4 — energy-filtered BSE detector (2 & 5 keV)
`bse_filter_contrast.png`. Restricting the BSE window from ">50 eV" to true
high-energy backscatter (>0.5–0.75·E₀) **roughly doubles** the dark-γ′ contrast:
−3.7%→−5.9% (2 keV) and −4.7%→−8.6% (5 keV), reaching/exceeding the Heinrich
limit. → A real energy-selective BSE detector sees a much stronger Z-contrast than
the arbitrary >50 eV cut used in the earlier decks (which dilutes it with
low-energy stopping-dominated escapes).

## Conclusion
The strong, scanning-enhanced dark-γ′ is best explained by a **γ′-selective
surface layer that builds with the beam** — carbon contamination and, more
potently, **Al/Ti-oxide on the Al/Ti-rich γ′** (Exp 3) — *not* by the intrinsic
alloy BSE contrast, which is weak at low landing energy (Exp 1) and saturates
under uniform carbon (Exp 2). Separately, imaging with an **energy-filtered BSE
detector** (Exp 4) or at **higher landing energy** (Exp 1) recovers the true
compositional Z-contrast (−6 to −9%), several times larger than the >50 eV-cut
decks report.
