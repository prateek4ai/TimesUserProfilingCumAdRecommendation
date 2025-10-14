# 📦 Publishing to PyPI

## Setup

Install build tools
pip install build twine
Create PyPI account
Visit: https://pypi.org/account/register/
text

## Build Package

Build distribution
python -m build
Verify
ls dist/
Should see: times_ctr_optimizer-1.0.0.tar.gz and .whl
text

## Test on TestPyPI (Recommended)

Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
Test install
pip install --index-url https://test.pypi.org/simple/ times-ctr-optimizer
text

## Publish to PyPI

Upload to PyPI
python -m twine upload dist/*
Verify
pip install times-ctr-optimizer
text

## Usage After Installation

Install package
pip install times-ctr-optimizer
Use in code
from times_ctr_optimizer import CTRPredictor
predictor = CTRPredictor(model_path="model.pt")
ctr = predictor.predict(user_id=123, item_id=456)
print(f"CTR: {ctr:.2%}")
text

## CLI Usage

After installation
times-ctr --user-id 123 --item-id 456 --model model.pt
text

---

**Author:** Prateek (IIT Patna MTech AI)
