"""Map top-level python import names to the PyPI package names that provide them."""

# import name -> pypi package name, only where the two differ (e.g. `import PIL` needs
# `pip install Pillow`). the analysis works with import names, this maps them for installs.
IMPORT_TO_PIP: dict[str, str] = {
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "dotenv": "python-dotenv",
    "fitz": "PyMuPDF",
    "jwt": "PyJWT",
    "OpenSSL": "pyOpenSSL",
    "PIL": "Pillow",
    "pptx": "python-pptx",
    "serial": "pyserial",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


def pip_package(
    import_name: str,
) -> str:
    """
    Find the PyPI package name for a top-level import name.

    Returns the mapped package name, or the import name unchanged when they are the same.
    """
    return IMPORT_TO_PIP.get(import_name, import_name)
