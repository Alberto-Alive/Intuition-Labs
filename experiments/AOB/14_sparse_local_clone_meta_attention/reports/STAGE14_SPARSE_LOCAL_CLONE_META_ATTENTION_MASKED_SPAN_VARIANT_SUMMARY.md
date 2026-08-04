# Stage 14b Masked-Span Variant Summary

| Variant | Joint PPL | Best Single Clone PPL | Joint Beats Best Clone | Clone Cosine | Mean Max Meta Weight |
| --- | ---: | ---: | --- | ---: | ---: |
| uniform | 8.5558 | 1076.3519 | True | 0.449747 | 0.6789 |
| boundary | 2.7246 | 212471.0710 | True | 0.313757 | 0.7111 |
| center | 1.7364 | 941616.6057 | True | 0.343509 | 0.6803 |

## Meta-Attention Mean Weights

- uniform: `[0.2442, 0.2620, 0.2732, 0.2206]`
- boundary: `[0.1900, 0.3151, 0.3125, 0.1823]`
- center: `[0.1585, 0.3254, 0.3568, 0.1593]`

## Interpretation

- `uniform` is the cleanest general-purpose fix to Stage 14's recency monopoly. The masked target is spread across all windows, clone cosine drops sharply, and meta-attention stays broadly balanced instead of collapsing onto one clone.
- `boundary` gives the strongest specialization signal and much better joint perplexity by forcing spans across clone boundaries, but it no longer probes the last window and therefore changes the target distribution materially.
- `center` is the easiest reconstruction variant in this sweep. It concentrates target spans on the middle boundary, which yields the best held-out perplexity here but also makes the task less general than `uniform`.
