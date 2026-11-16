Results from commit `f521303`: 50 trials each, random position and rotation, fixed seeds.
Detection error is mean / max. Original = RGB thresholds, no blur, whole-image average, no grasp check.

| Test | Original: success | Original: detection error | Current: success | Current: detection error |
|---|---|---|---|---|
| Normal conditions | 100% | 1.61 / 2.28 mm | 100% | 1.63 / 2.30 mm |
| Camera noise, sigma = 40 | 2% | 58.43 / 85.77 mm | 100% | 1.66 / 2.29 mm |
| Camera noise, sigma = 60 | 0% | 150.56 / 241.43 mm | 100% | 1.70 / 2.53 mm |
| Dim light (x0.2) | 0% | cube not found | 100% | 1.71 / 2.17 mm |
| Bright light (x2.5) | 56% | 18.28 / 29.14 mm | 100% | 3.33 / 18.94 mm |
| Blue / green / orange distractors | 100% | 1.62 / 2.28 mm | 100% | 1.63 / 2.26 mm |
| Second red cube, >= 10 cm away | 0% | 116.34 / 226.18 mm | 100% | 1.62 / 2.02 mm |
| Second red cube, 10 mm gap | 0% | 28.93 / 29.96 mm | 100% | 1.62 / 2.30 mm |
| Second red cube, 2 mm gap | 0% | 24.91 / 26.00 mm | 96% | 2.55 / 24.79 mm |
