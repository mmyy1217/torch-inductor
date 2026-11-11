# Run116 full matrix result

26/26 correctness checks and156/156 timing receipts passed. Two rounds, three backends,30 raw NPU-event samples per receipt. API-path metric on shared Ascend950PR; not exclusive pure-kernel timing.

TileLang faster than Triton:26/26 (1.28–5.49x). Faster than native FAS V4:0/26. Goal remains unmet.

| Case | TileLang us | Triton us | FAS V4 us | FAS / TileLang |
|---|---:|---:|---:|---:|
| d128-small-mha | 58.424 | 77.848 | 42.070 | 0.720 |
| d128-mha | 60.420 | 154.339 | 54.954 | 0.910 |
| d128-gqa | 60.159 | 155.875 | 52.599 | 0.874 |
| d128-mqa64 | 177.692 | 636.599 | 134.155 | 0.755 |
| d128-batch2-unequal | 74.201 | 248.404 | 64.563 | 0.870 |
| d128-batch4-unequal | 69.524 | 138.078 | 52.271 | 0.752 |
| d128-long-mqa | 262.645 | 1249.721 | 252.018 | 0.960 |
| d128-single-kv | 59.588 | 83.137 | 55.997 | 0.940 |
| d256-small-mha | 55.835 | 81.852 | 31.898 | 0.571 |
| d256-mha | 95.956 | 201.409 | 72.157 | 0.752 |
| d256-gqa | 89.383 | 194.608 | 77.034 | 0.862 |
| d256-mqa64 | 386.154 | 903.090 | 296.382 | 0.768 |
| d256-batch2-unequal | 138.592 | 325.311 | 120.516 | 0.870 |
| d256-batch4-unequal | 93.296 | 181.869 | 68.085 | 0.730 |
| d256-long-mqa | 628.421 | 1795.272 | 564.151 | 0.898 |
| d256-single-kv | 60.631 | 84.248 | 32.518 | 0.536 |
| d512-small-mha | 56.159 | 73.619 | 47.913 | 0.853 |
| d512-mha | 180.500 | 270.451 | 164.161 | 0.909 |
| d512-gqa | 174.382 | 267.057 | 163.079 | 0.935 |
| d512-mqa64 | 1092.295 | 1535.263 | 817.461 | 0.748 |
| d512-batch2-unequal | 296.718 | 460.897 | 282.546 | 0.952 |
| d512-batch4-unequal | 193.589 | 261.013 | 143.078 | 0.739 |
| d512-long-mqa | 1926.229 | 3055.872 | 1626.528 | 0.844 |
| d512-single-kv | 56.847 | 72.615 | 44.648 | 0.785 |
| d128-long-gqa | 839.642 | 4605.582 | 830.264 | 0.989 |
| d512-long-mha | 1032.990 | 1649.153 | 838.840 | 0.812 |

Numbers are means of the two round medians. Speedup >1 means TileLang faster. Interrupted attempts excluded by monitor. PR8192 HQ32/HKV1 is a separate run115 case, not this matrix. Pending candidates117/121/122 have no device validation.
