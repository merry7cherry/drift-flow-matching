# Software checks

Tested CPU environment: Python 3.11.15, PyTorch 2.8.0, torchvision 0.23.0, NumPy 2.0.2 and Matplotlib 3.9.4 on macOS arm64. Package versions are recorded in the [image constraints](../image_generation/constraints-tested.txt) and [trajectory dependencies](../generation_trajectory/pyproject.toml).

After installing both modules with the image module's `eval`, `dev` and `data` extras, run from the repository root:

```bash
python -m pytest image_generation/tests -q
python -m unittest discover -s generation_trajectory/tests -v
python scripts/check_site.py
python -m unittest discover -s scripts -p "test_*.py" -v
```

The [CPU checks workflow](https://github.com/merry7cherry/drift-flow-matching/actions/workflows/checks.yml) runs these commands on Ubuntu. See [figure validation](assets.md#validation) to check against a local manuscript checkout.
