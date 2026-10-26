| job | N repos | wall-clock | peak RSS | commit |
|---|---|---|---|---|
| CF-only topfilter_half | 644 | 14.0 s | 14 MB | 7c763fa |
| CF-only preprocessing | 5,954 | 9.5 s | 329 MB | 7c763fa |
| CF-only make_folds | 5,954 | 0.0 s | 16 MB | 7c763fa |
| CF-only topfilter_half | 5,954 | 956.1 s | 27 MB | 7c763fa |
| CF-only `outpath/cfonly_25k` | not run | | | |
| hybrid cnbn/meta_only | 644 | 35.2 s | | c3eb693 |
| hybrid cnbn/readme_8k | 644 | 367.9 s | | c3eb693 |
| hybrid cnbn/readme_8k_desc20 | 644 | 333.7 s | | c3eb693 |
| hybrid topfilter/meta_only/A | 644 | 21.8 s | | c3eb693 |
| hybrid topfilter/meta_only/B | 644 | 26.5 s | | c3eb693 |
| hybrid topfilter/meta_only/C | 644 | 20.7 s | | c3eb693 |
| hybrid topfilter/readme_8k/A | 644 | 20.3 s | | c3eb693 |
| hybrid topfilter/readme_8k/B | 644 | 23.4 s | | c3eb693 |
| hybrid topfilter/readme_8k/C | 644 | 23.7 s | | c3eb693 |
| hybrid topfilter/readme_8k_desc20/A | 644 | 20.6 s | | c3eb693 |
| hybrid topfilter/readme_8k_desc20/B | 644 | 23.4 s | | c3eb693 |
| hybrid topfilter/readme_8k_desc20/C | 644 | 21.9 s | | c3eb693 |
| hybrid `outpath/hybrid_25k/experiments` | not run | | | |
| README fetch 2026-09-25 | 10,000 | 7,722 s (0.77 s/repo, 4,662/h) | | |
| README fetch 2026-10-02 | 6,585 | 4,839 s (0.73 s/repo, 4,898/h) | | |

TopFilter scaling: t = 14.0 s * (N / 644)^1.90  (fitted on N=644 and N=5,954; the N^2 model predicts 1,195 s at N=5,954, measured 956 s)
  projection N=10,000: 0.7 h
  projection N=15,000: 1.5 h
  projection N=25,000: 4.1 h
