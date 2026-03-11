# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import socket
import sys

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.realpath(os.path.join(_PATH_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_PATH_ROOT, "src"))

import litlogger  # noqa: E402

# -- Project information -----------------------------------------------------

project = "litlogger"
copyright = "Lightning AI"  # noqa: A001
author = "Lightning AI"
version = litlogger.__version__
release = litlogger.__version__

# -- General configuration ---------------------------------------------------

needs_sphinx = "8.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx_togglebutton",
    "myst_parser",
]

templates_path = ["_templates"]

# myst-parser settings
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "_templates"]
pygments_style = None

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/Lightning-AI/litlogger",
}

# html_favicon = "_static/images/icon.svg"  # TODO: add favicon
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = project + "-doc"

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------


def _can_resolve_host(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443)
    except OSError:
        return False
    return True


if all(_can_resolve_host(host) for host in ("docs.python.org", "pytorch.org", "lightning.ai")):
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "torch": ("https://pytorch.org/docs/stable/", None),
        "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
        "lightning.fabric": ("https://lightning.ai/docs/fabric/stable/", None),
    }
else:
    intersphinx_mapping = {}
nitpicky = True

nitpick_ignore = [
    # External Lightning/PyTorch — mocked, can't resolve
    ("py:class", "pytorch_lightning.loggers.LitLogger"),
    ("py:class", "lightning.pytorch.loggers.LitLogger"),
    ("py:class", "lightning.pytorch.loggers.litlogger.LitLogger"),
    ("py:class", "lightning.fabric.loggers.logger.Logger"),
    ("py:class", "lightning.fabric.utilities.types._PATH"),
    # lightning_sdk — mocked
    ("py:class", "Teamspace"),
    ("py:class", "lightning_sdk.Teamspace"),
    ("py:class", "lightning_sdk.teamspace.Teamspace"),
    # Internal type aliases
    ("py:class", "_PATH"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Optional"),
    ("py:class", "pathlib.Path"),
    ("py:class", "enum.Enum"),
    # typing.Union — inherited from LitLogger parent class, role mismatch (py:data vs py:class)
    ("py:data", "typing.Union"),
]

# -- Options for autodoc -----------------------------------------------------

autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_typehints = "description"
typehints_description_target = "documented_params"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

# True to prefix each section label with the name of the document it is in
autosectionlabel_prefix_document = True

# -- Options for doctest -----------------------------------------------------

doctest_test_doctest_blocks = ""
doctest_global_setup = """
import os
import litlogger
"""

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = ">>> "
copybutton_prompt_text1 = "... "
copybutton_only_copy_prompt_lines = True

# -- Mock imports for optional dependencies ----------------------------------

autodoc_mock_imports = [
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "lightning_sdk",
    "litmodels",
    "blake3",
    "psutil",
    "rich",
    "protobuf",
    "torch",
]


# -- Options for linkcheck ---------------------------------------------------

linkcheck_anchors = False
linkcheck_timeout = 60
linkcheck_retries = 3
linkcheck_ignore = [
    r"https://lightning\.ai/.*",  # requires auth
]


def setup(app) -> None:  # noqa: ANN001
    app.add_css_file("main.css")
