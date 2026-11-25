# Gosha's handoff to the team

Scope: Evaluation Protocol, Results, Reproducibility and discussion material.
This is an insertable contribution, not the full six-page report or a slide
deck. The team still needs to agree on the shared Abstract, Introduction,
Discussion, Conclusion, contribution wording and AI-use statement.

## To Masha: assemble the report

Copy this directory and include `evaluation_results.tex`. It has no
`documentclass`, preamble or bibliography command. Required packages are
`amsmath`, `amssymb`, `booktabs`, `graphicx` and `natbib`. Merge the entries from
`references.bib` into the shared bibliography, avoiding duplicate keys.

If compiling the main report from the repository root:

```tex
\newcommand{\goshaRoot}{report/gosha}
\input{report/gosha/evaluation_results.tex}
```

If the main `.tex` file is compiled from `report/`, use `gosha` as the root
instead. Labels are prefixed `gosha-` to reduce collisions. The fragment loads
generated `numbers.tex` and the two main tables. The full-width `table*`
environments can be adapted to the final layout without editing values.

Main assets:

- `preview.pdf`: requested two-column integration preview, two pages.
- `preview_neurips.pdf`: single-column formatting reference, two pages of
  contribution text plus a references-only third page. The standard
  NeurIPS style is not a two-column conference format.
- `tables/main_results.tex` and `tables/tie_seed_summary.tex`.
- `figures/target_ties.pdf`: used in the main fragment.
- `figures/gap_comparison.pdf`: optional discussion/supplement figure; its
  caption must distinguish paper RANDOM (five-run mean) from analytic RANDOM.
- `figures/gap_comparison.tex`: ready-to-use caption for that optional figure.
- `contribution.tex` and `ai_use.tex`: factual draft snippets for team review.

`tables/head_tail_results.tex` is optional detailed material. Full-precision
head/tail and seed CSVs are separate to keep the main contribution short.
Check final float placement and page count after combining all authors' text.

## To Dasha: presentation inputs

Use the PDF figures for vector slides or the matching PNGs. The tie-frequency
figure labels exact count/total and percentages, with a linear percentage
axis. The gap-comparison figure uses a logarithmic axis and preserves the
small nonzero measured gaps. It deliberately omits a paper RotatE gap: the
published equal three-decimal MRR values do not resolve its unrounded gap.

Key message: tie-breaking has negligible impact on these two budget
checkpoints; the large published ConvKB effect was not reproduced. This does
not refute the paper or identify a cause for the difference. Seed variability
is evaluation randomness, not training uncertainty. Full scoring ran earlier
on T4; this handoff's new CPU checks do not repeat that full GPU experiment.

Suggested speaking text for Gosha's segment only:

> We evaluated each frozen checkpoint on forty thousand nine hundred and
> thirty-two head and tail queries. We filtered other known correct answers
> and defined ties by exact equality of raw float32 scores. All ranking
> policies use the same saved counts. Expected random MRR averages reciprocal
> ranks over the target's tie group; five evaluation seeds provide a separate
> check. We found twenty-six tied queries for RotatE and three for corrected
> ConvKB. The optimistic advantage was approximately three times ten to the
> minus eight and six times ten to the minus eleven MRR units, respectively.
> Thus, the large ConvKB effect reported in the paper was not reproduced on
> our budget checkpoints. We cannot attribute the difference to a particular
> training or implementation choice without further experiments. Our tables,
> figures and independent checks can be reconstructed from saved counts
> without weights or a GPU.

The segment's timing and the full eight-minute allocation remain for team
agreement. No presentation or team approval is claimed by this handoff.
