# Model Card — SVHN House Number Recognition

## Intended Use
Digit recognition for house numbers captured from street-view imagery to support delivery robots.

## Training Data
SVHN (train + optional extra). See dataset license.

## Metrics
Report Top-1 accuracy/NLL on official test split. Include seeds and variance if possible.

## Ethical Considerations & Limitations
- SVHN is digits only; real addresses may include multi-digit sequences and occlusions.
- Domain shift: other countries/plates/fonts may degrade accuracy.
- Not suitable for sensitive surveillance use.

## Risks & Mitigations
- False recognition → route errors. Use confidence thresholds and fallback strategies.

## License
MIT (this repo). Dataset per its own license.
