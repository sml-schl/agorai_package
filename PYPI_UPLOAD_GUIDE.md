# PyPI Upload Guide for AgorAI

This guide provides step-by-step instructions for uploading the AgorAI package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - PyPI (production): https://pypi.org/account/register/
   - TestPyPI (testing): https://test.pypi.org/account/register/

2. **API Tokens**: Generate API tokens for authentication:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

3. **Required Tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Step 1: Prepare the Package

### 1.1 Verify Package Structure

Ensure your package structure matches:

```
agorai-package/
├── src/
│   └── agorai/
│       ├── __init__.py
│       ├── aggregate/
│       ├── synthesis/
│       └── bias/
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
└── docs/
```

### 1.2 Update Version Number

Edit `pyproject.toml` to update the version:

```toml
[project]
name = "agorai"
version = "0.1.0"  # Update this for each release
```

Version numbering follows [Semantic Versioning](https://semver.org/):
- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features (backward compatible)
- `1.0.0` - Stable release

### 1.3 Update README and Documentation

Ensure README.md is comprehensive and includes:
- Installation instructions
- Quick start examples
- API overview
- Links to full documentation

## Step 2: Build the Package

### 2.1 Clean Previous Builds

```bash
cd /Users/workandstudy/Desktop/Masterarbeit/Research/Experiment/agorai-package
rm -rf dist/ build/ *.egg-info src/*.egg-info
```

### 2.2 Build Distribution Files

```bash
python -m build
```

This creates two files in `dist/`:
- `agorai-0.1.0.tar.gz` (source distribution)
- `agorai-0.1.0-py3-none-any.whl` (wheel distribution)

### 2.3 Verify Build Contents

```bash
tar -tzf dist/agorai-0.1.0.tar.gz | head -20
```

Check that all necessary files are included.

## Step 3: Test Upload to TestPyPI

### 3.1 Configure TestPyPI Credentials

Create or edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Important:** Replace `pypi-YOUR_*_TOKEN_HERE` with your actual API tokens.

Set secure permissions:
```bash
chmod 600 ~/.pypirc
```

### 3.2 Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Expected output:
```
Uploading distributions to https://test.pypi.org/legacy/
Uploading agorai-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 kB • 00:00 • ?
Uploading agorai-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.0/45.0 kB • 00:00 • ?

View at:
https://test.pypi.org/project/agorai/0.1.0/
```

### 3.3 Test Installation from TestPyPI

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agorai

# Test basic functionality
python -c "from agorai.aggregate import aggregate; print(aggregate([[1,0],[1,0],[0,1]], method='majority'))"
```

### 3.4 Test All Modules

```bash
python << EOF
from agorai.aggregate import aggregate, list_methods
from agorai.synthesis import Agent, synthesize
from agorai.bias import mitigate_bias, BiasConfig

# Test aggregate
print("Testing aggregate...")
result = aggregate([[0.8, 0.2], [0.3, 0.7]], method="atkinson", epsilon=1.0)
print(f"Aggregate test passed: {result['winner']}")

# Test synthesis (without API keys - will show graceful error)
print("\nTesting synthesis...")
try:
    agent = Agent("ollama", "llama3.2", base_url="http://localhost:11434")
    print("Synthesis module loaded successfully")
except Exception as e:
    print(f"Synthesis test (expected without Ollama): {e}")

# Test bias
print("\nTesting bias...")
config = BiasConfig(providers=["ollama"])
print("Bias module loaded successfully")

print("\n✅ All modules loaded successfully!")
EOF
```

### 3.5 Clean Up Test Environment

```bash
deactivate
rm -rf test_env
```

## Step 4: Upload to Production PyPI

### 4.1 Final Checks

- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Version number is correct
- [ ] CHANGELOG.md is updated (if exists)
- [ ] Git repository is tagged with version

### 4.2 Upload to PyPI

```bash
python -m twine upload dist/*
```

Expected output:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading agorai-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 kB • 00:00 • ?
Uploading agorai-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.0/45.0 kB • 00:00 • ?

View at:
https://pypi.org/project/agorai/0.1.0/
```

### 4.3 Verify Installation

```bash
pip install agorai
```

Visit the package page: https://pypi.org/project/agorai/

## Step 5: Post-Release Tasks

### 5.1 Tag the Release in Git

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

### 5.2 Create GitHub Release (if using GitHub)

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Select the tag `v0.1.0`
4. Add release notes describing changes
5. Attach the distribution files from `dist/`

### 5.3 Update Documentation

- Update installation instructions
- Add release notes to CHANGELOG.md
- Update README.md with any new examples

## Common Issues and Solutions

### Issue 1: "File already exists"

**Error:**
```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists
```

**Solution:** You cannot re-upload the same version. Increment the version number in `pyproject.toml` and rebuild.

### Issue 2: "Invalid distribution metadata"

**Error:**
```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
Invalid distribution metadata
```

**Solution:** Check `pyproject.toml` for syntax errors or missing required fields. Validate with:
```bash
pip install validate-pyproject
validate-pyproject pyproject.toml
```

### Issue 3: "Package name already taken"

**Error:**
```
HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/
The name 'agorai' is already registered
```

**Solution:** Choose a different package name in `pyproject.toml`. Common patterns:
- `agorai-core`
- `agorai-ai`
- `your-organization-agorai`

### Issue 4: Authentication failure

**Error:**
```
HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/
Invalid or non-existent authentication information
```

**Solution:**
1. Verify API token is correct in `~/.pypirc`
2. Ensure token has "Upload packages" scope
3. Try using `--username __token__ --password YOUR_TOKEN` directly:
   ```bash
   python -m twine upload --username __token__ --password YOUR_TOKEN dist/*
   ```

### Issue 5: Missing dependencies

**Error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solution:** Ensure all dependencies are listed in `pyproject.toml` under `dependencies`:
```toml
dependencies = [
    "numpy>=1.20.0",
    "requests>=2.25.0",
]
```

## Best Practices

### 1. Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Never delete a release from PyPI
- Use pre-release versions for testing: `0.1.0a1`, `0.1.0b1`, `0.1.0rc1`

### 2. Testing Strategy

1. Test locally with `pip install -e .`
2. Test on TestPyPI before production
3. Create virtual environments for testing
4. Test all optional dependencies: `pip install agorai[all]`

### 3. Documentation

- Keep README.md concise and focused on quick start
- Maintain detailed docs in separate files
- Include code examples that work out-of-the-box
- Document all public APIs with docstrings

### 4. Security

- Never commit API tokens to version control
- Use `.gitignore` to exclude sensitive files
- Rotate tokens periodically
- Use separate tokens for CI/CD vs. manual uploads

### 5. Release Checklist

Before each release:

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Build package: `python -m build`
- [ ] Test on TestPyPI
- [ ] Upload to PyPI
- [ ] Tag release in Git
- [ ] Create GitHub release
- [ ] Verify installation: `pip install agorai`

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to repository secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Add new repository secret: `PYPI_API_TOKEN`
3. Paste your PyPI API token

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [PEP 517/518](https://www.python.org/dev/peps/pep-0517/) (Modern packaging)

## Support

For issues with:
- **Package structure/setup:** Check [Python Packaging Guide](https://packaging.python.org/tutorials/packaging-projects/)
- **PyPI upload errors:** Review [PyPI Help](https://pypi.org/help/)
- **Build errors:** Consult [setuptools documentation](https://setuptools.pypa.io/)
