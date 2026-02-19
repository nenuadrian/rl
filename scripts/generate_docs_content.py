#!/usr/bin/env python3
"""Generate MkDocs content from project sources.

This script populates docs/generated with:
- Latest report markdown/assets from reports/latest
- Hyperparameter source snapshots from hyperparameters/*.py
- API-style markdown reference for all modules under trainers/
- Math-annotated pages for trainer modules that contain `# LaTeX:` comments
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "generated"
REPORTS_LATEST_DIR = ROOT / "reports" / "latest"
HYPERPARAMETERS_DIR = ROOT / "hyperparameters"
TRAINERS_DIR = ROOT / "minerva/trainers"

MISSING = object()
LATEX_INLINE_PATTERN = re.compile(
    r"^(?P<code>.*?)(?:\s+#\s*LaTeX:\s*(?P<formula>.+))\s*$"
)
LATEX_STANDALONE_PATTERN = re.compile(r"^\s*#\s*LaTeX:\s*(?P<formula>.+?)\s*$")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def clean_generated_dir() -> None:
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def generate_report_docs() -> None:
    target_dir = GENERATED_DIR / "report"
    target_dir.mkdir(parents=True, exist_ok=True)

    readme_path = REPORTS_LATEST_DIR / "README.md"
    if not readme_path.exists():
        write_text(
            target_dir / "latest.md",
            "# Latest Report\n\n"
            "No `reports/latest/README.md` file was found at build time.\n",
        )
        return

    for source in REPORTS_LATEST_DIR.rglob("*"):
        if source.is_dir():
            continue
        rel = source.relative_to(REPORTS_LATEST_DIR)
        if rel.as_posix() == "README.md":
            continue
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    report_markdown = readme_path.read_text(encoding="utf-8")
    page_content = (
        "# Latest Experiment Report\n\n"
        f"_Source: `{readme_path.relative_to(ROOT).as_posix()}`_\n\n"
        f"{report_markdown}\n"
    )
    write_text(target_dir / "latest.md", page_content)


def format_arg(arg: ast.arg, default: object = MISSING) -> str:
    result = arg.arg
    if arg.annotation is not None:
        result += f": {ast.unparse(arg.annotation)}"
    if default is not MISSING:
        result += f" = {ast.unparse(default)}"
    return result


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[str] = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = list(node.args.defaults)
    defaults_start = len(positional) - len(defaults)

    for index, arg in enumerate(positional):
        default = defaults[index - defaults_start] if index >= defaults_start else MISSING
        parts.append(format_arg(arg, default))
        if node.args.posonlyargs and index == len(node.args.posonlyargs) - 1:
            parts.append("/")

    if node.args.vararg is not None:
        parts.append(f"*{format_arg(node.args.vararg)}")
    elif node.args.kwonlyargs:
        parts.append("*")

    for kwarg, kwdefault in zip(node.args.kwonlyargs, node.args.kw_defaults):
        default = MISSING if kwdefault is None else kwdefault
        parts.append(format_arg(kwarg, default))

    if node.args.kwarg is not None:
        parts.append(f"**{format_arg(node.args.kwarg)}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    result = f"{prefix} {node.name}({', '.join(parts)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def module_name_from_path(base_dir: Path, file_path: Path, root_name: str) -> str:
    rel = file_path.relative_to(base_dir)
    parts = list(rel.parts)
    if not parts:
        return root_name
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    if not parts:
        return root_name
    return f"{root_name}.{'.'.join(parts)}"


def ordered_python_files(base_dir: Path) -> Iterable[Path]:
    return sorted(path for path in base_dir.rglob("*.py") if path.is_file())


def generate_hyperparameter_docs() -> None:
    section_dir = GENERATED_DIR / "hyperparameters"
    files_dir = section_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "# Hyperparameters Source",
        "",
        "These pages embed the exact source code from `hyperparameters/*.py`.",
        "",
    ]

    py_files = ordered_python_files(HYPERPARAMETERS_DIR)
    if not py_files:
        index_lines.append("No Python files were found in `hyperparameters/`.")
        write_text(section_dir / "index.md", "\n".join(index_lines) + "\n")
        return

    for path in py_files:
        rel = path.relative_to(ROOT).as_posix()
        stem = path.relative_to(HYPERPARAMETERS_DIR).as_posix().removesuffix(".py")
        slug = stem.replace("/", "_")
        page_rel = f"files/{slug}.md"
        index_lines.append(f"- [`{rel}`]({page_rel})")

        source = path.read_text(encoding="utf-8")
        page = [
            f"# `{rel}`",
            "",
            "```python",
            source.rstrip(),
            "```",
            "",
        ]
        write_text(section_dir / page_rel, "\n".join(page))

    write_text(section_dir / "index.md", "\n".join(index_lines) + "\n")


def docs_for_module(file_path: Path, module_name: str) -> str:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    module_doc = ast.get_docstring(tree, clean=True)

    lines = [
        f"# `{module_name}`",
        "",
        f"_Source: `{file_path.relative_to(ROOT).as_posix()}`_",
        "",
    ]

    if module_doc:
        lines.extend(["## Module Docstring", "", module_doc, ""])

    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    if functions:
        lines.extend(["## Functions", ""])
        for fn in functions:
            lines.extend([f"### `{fn.name}`", "", "```python", format_signature(fn), "```", ""])
            fn_doc = ast.get_docstring(fn, clean=True)
            lines.append(fn_doc if fn_doc else "_No docstring provided._")
            lines.append("")

    if classes:
        lines.extend(["## Classes", ""])
        for cls in classes:
            lines.append(f"### `{cls.name}`")
            lines.append("")
            if cls.bases:
                bases = ", ".join(ast.unparse(base) for base in cls.bases)
                lines.append(f"Base classes: `{bases}`")
                lines.append("")

            cls_doc = ast.get_docstring(cls, clean=True)
            lines.append(cls_doc if cls_doc else "_No docstring provided._")
            lines.append("")

            methods = [
                n
                for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                lines.extend([f"#### Methods in `{cls.name}`", ""])
            for method in methods:
                lines.extend(
                    [
                        f"##### `{method.name}`",
                        "",
                        "```python",
                        format_signature(method),
                        "```",
                        "",
                    ]
                )
                method_doc = ast.get_docstring(method, clean=True)
                lines.append(method_doc if method_doc else "_No docstring provided._")
                lines.append("")

    if not functions and not classes:
        lines.extend(["No top-level classes or functions found.", ""])

    return "\n".join(lines)


def extract_latex_annotations(source: str) -> list[tuple[int, str, str]]:
    annotations: list[tuple[int, str, str]] = []
    pending_formula: str | None = None

    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        standalone_match = LATEX_STANDALONE_PATTERN.match(raw_line)
        if standalone_match is not None:
            pending_formula = standalone_match.group("formula").strip()
            continue

        inline_match = LATEX_INLINE_PATTERN.match(raw_line)
        if inline_match is not None and inline_match.group("formula") is not None:
            code = inline_match.group("code").rstrip()
            formula = inline_match.group("formula").strip()
            if code:
                annotations.append((line_no, code, formula))
            pending_formula = None
            continue

        stripped = raw_line.strip()
        if pending_formula and stripped and not stripped.startswith("#"):
            annotations.append((line_no, raw_line.rstrip(), pending_formula))
            pending_formula = None
        elif stripped:
            pending_formula = None

    return annotations


def docs_for_trainer_math(file_path: Path, module_name: str) -> str:
    source = file_path.read_text(encoding="utf-8")
    annotations = extract_latex_annotations(source)
    rel_path = file_path.relative_to(ROOT).as_posix()

    lines = [
        f"# `{module_name}` Math-Annotated Source",
        "",
        f"_Source: `{rel_path}`_",
        "",
        "Each `# LaTeX:` annotation is rendered below next to its source line.",
        "",
    ]

    if not annotations:
        lines.extend(["No `# LaTeX:` annotations were found in this file.", ""])
    else:
        lines.extend(["## Rendered Math Annotations", ""])
        for line_no, code, formula in annotations:
            lines.extend(
                [
                    f"### Line {line_no}",
                    "",
                    "```python",
                    code,
                    "```",
                    "",
                    "$$",
                    formula,
                    "$$",
                    "",
                ]
            )

    lines.extend(["## Full Source", "", "```python", source.rstrip(), "```", ""])
    return "\n".join(lines)


def generate_trainers_api_docs() -> None:
    section_dir = GENERATED_DIR / "trainers-api"
    section_dir.mkdir(parents=True, exist_ok=True)

    py_files = ordered_python_files(TRAINERS_DIR)
    if not py_files:
        write_text(
            section_dir / "index.md",
            "# Trainers API\n\nNo Python files were found in `trainers/`.\n",
        )
        return

    index_lines = [
        "# Trainers API Reference",
        "",
        "Auto-generated reference for all modules in `trainers/`.",
        "",
    ]

    for path in py_files:
        module_name = module_name_from_path(TRAINERS_DIR, path, "trainers")
        rel_module = module_name.replace(".", "/")
        page = section_dir / f"{rel_module}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        write_text(page, docs_for_module(path, module_name))

        rel_link = page.relative_to(section_dir).as_posix()
        index_lines.append(f"- [`{module_name}`]({rel_link})")

    write_text(section_dir / "index.md", "\n".join(index_lines) + "\n")


def generate_trainers_math_docs() -> None:
    section_dir = GENERATED_DIR / "trainers-math"
    section_dir.mkdir(parents=True, exist_ok=True)

    trainer_files = sorted(
        path
        for path in ordered_python_files(TRAINERS_DIR)
        if path.name == "trainer.py"
    )
    extra_annotated_files = []
    for path in ordered_python_files(TRAINERS_DIR):
        if path.name == "trainer.py":
            continue
        source = path.read_text(encoding="utf-8")
        if extract_latex_annotations(source):
            extra_annotated_files.append(path)

    math_files = trainer_files + extra_annotated_files
    if not math_files:
        write_text(
            section_dir / "index.md",
            "# Trainers Math\n\nNo trainer files were found in `trainers/**/trainer.py`.\n",
        )
        return

    index_lines = [
        "# Trainers Math-Annotated Source",
        "",
        "Trainer pages and annotated trainer modules that render inline `# LaTeX:` comments as formulas.",
        "",
    ]

    for path in math_files:
        module_name = module_name_from_path(TRAINERS_DIR, path, "trainers")
        rel_module = module_name.replace(".", "/")
        page = section_dir / f"{rel_module}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        write_text(page, docs_for_trainer_math(path, module_name))

        rel_link = page.relative_to(section_dir).as_posix()
        index_lines.append(f"- [`{module_name}`]({rel_link})")

    write_text(section_dir / "index.md", "\n".join(index_lines) + "\n")


def main() -> None:
    clean_generated_dir()
    generate_report_docs()
    generate_hyperparameter_docs()
    generate_trainers_api_docs()
    generate_trainers_math_docs()


if __name__ == "__main__":
    main()
