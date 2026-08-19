#!/usr/bin/env python3
"""Generate the API Reference tab from rustdoc JSON.

Usage (from the repository root):

    RUSTC_BOOTSTRAP=1 cargo rustdoc -p dspy-rs --lib --all-features -- -Z unstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc -p dsrs-tools --lib -- -Z unstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc -p dsrs_macros --lib -- -Z unstable-options --output-format json
    python3 docs/scripts/gen_api.py

Reads target/doc/{dspy_rs,dsrs_tools,dsrs_macros}.json and rewrites
docs/docs/api/*.mdx. Every page is fully generated; do not edit them by
hand. The item inventory and doc summaries come from the compiler, so
the pages cannot drift from the code. Full signatures, methods, and
long-form docs live on docs.rs; every item links to its docs.rs page.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC_JSON = REPO / "target" / "doc"
OUT = REPO / "docs" / "docs" / "api"

CRATES = [
    # (json file, crate name on docs.rs, rustdoc path root, page stem for extra crates)
    ("dspy_rs.json", "dspy-rs", "dspy_rs", None),
    ("dsrs_tools.json", "dsrs-tools", "dsrs_tools", "dsrs-tools"),
    ("dsrs_macros.json", "dsrs-macros", "dsrs_macros", "dsrs-macros"),
]

KIND_LABELS = [
    ("use", "Re-exports"),
    ("module", "Modules"),
    ("struct", "Structs"),
    ("enum", "Enums"),
    ("trait", "Traits"),
    ("function", "Functions"),
    ("type_alias", "Type aliases"),
    ("constant", "Constants"),
    ("static", "Statics"),
    ("macro", "Macros"),
    ("proc_macro", "Proc macros"),
]

URL_PREFIX = {
    "struct": "struct",
    "enum": "enum",
    "trait": "trait",
    "function": "fn",
    "type_alias": "type",
    "constant": "constant",
    "static": "static",
    "macro": "macro",
}


def first_sentence(docs: str) -> str:
    """First paragraph of a doc comment, flattened for a table cell."""
    if not docs:
        return ""
    para = docs.strip().split("\n\n")[0]
    para = " ".join(line.strip() for line in para.splitlines())
    # [`X`](url) and [`X`] -> `X`
    para = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", para)
    para = re.sub(r"\[([^\]]+)\]", r"\1", para)
    para = para.replace("|", "\\|")
    if len(para) > 180:
        cut = para[:180]
        if "." in cut:
            cut = cut[: cut.rindex(".") + 1]
        else:
            cut = cut.rstrip() + "..."
        para = cut
    return para


class Crate:
    def __init__(self, data, docsrs_name, path_root):
        self.index = data["index"]
        self.paths = data["paths"]
        self.root = str(data["root"])
        self.version = data.get("crate_version") or ""
        self.docsrs = docsrs_name
        self.path_root = path_root

    def item(self, iid):
        return self.index.get(str(iid))

    def kind_of(self, item):
        return next(iter(item["inner"].keys())) if item.get("inner") else None

    def docsrs_url(self, iid, kind=None):
        """docs.rs URL for a local item id, resolved through the paths table."""
        p = self.paths.get(str(iid))
        if not p:
            return f"https://docs.rs/{self.docsrs}/latest/{self.path_root}/"
        segs = p["path"]
        kind = kind or p["kind"]
        base = f"https://docs.rs/{self.docsrs}/latest/" + "/".join(segs[:-1])
        name = segs[-1]
        if kind == "module":
            return f"{base}/{name}/index.html"
        if kind == "proc_macro":
            return f"{base}/macro.{name}.html"
        if kind in ("proc_attribute",):
            return f"{base}/attr.{name}.html"
        if kind in ("proc_derive",):
            return f"{base}/derive.{name}.html"
        prefix = URL_PREFIX.get(kind, kind)
        return f"{base}/{prefix}.{name}.html"

    def public_items(self, module_item):
        out = []
        for iid in module_item["inner"]["module"]["items"]:
            it = self.item(iid)
            if it is None:
                continue
            if it.get("visibility") not in ("public", "default"):
                continue
            out.append((iid, it))
        return out

    def proc_macro_kind(self, item):
        pm = item["inner"].get("proc_macro", {})
        return {"derive": "derive", "attr": "attribute", "bang": "macro"}.get(
            pm.get("kind"), "macro"
        )


def render_tables(crate: Crate, items, heading_level="##", path_note=""):
    """Group (id, item) pairs by kind and render one table per kind."""
    groups = {}
    for iid, it in items:
        kind = crate.kind_of(it)
        if kind in (None, "impl", "struct_field", "variant"):
            continue
        groups.setdefault(kind, []).append((iid, it))

    lines = []
    for kind, label in KIND_LABELS:
        if kind not in groups:
            continue
        rows = []
        for iid, it in groups[kind]:
            if kind == "use":
                use = it["inner"]["use"]
                name = use.get("name") or use.get("source", "")
                target = use.get("id")
                if use.get("is_glob") or use.get("glob"):
                    rows.append((f"`{use.get('source', name)}::*`", first_sentence(it.get("docs") or "Glob re-export."), None))
                    continue
                url = crate.docsrs_url(target) if target is not None else None
                src = use.get("source", "")
                doc = first_sentence(it.get("docs", "")) or f"Re-export of `{src}`."
                rows.append((f"`{name}`", doc, url))
            elif kind == "proc_macro":
                pk = crate.proc_macro_kind(it)
                url = crate.docsrs_url(iid)
                rows.append((f"`{it['name']}`", f"({pk}) " + first_sentence(it.get("docs", "")), url))
            else:
                url = crate.docsrs_url(iid, kind)
                rows.append((f"`{it['name']}`", first_sentence(it.get("docs", "")), url))
        if not rows:
            continue
        rows.sort(key=lambda r: r[0].lower())
        lines.append(f"{heading_level} {label}{path_note}")
        lines.append("")
        lines.append("| Item | Description |")
        lines.append("|---|---|")
        for name, doc, url in rows:
            cell = f"[{name}]({url})" if url else name
            lines.append(f"| {cell} | {doc} |")
        lines.append("")
    return lines


def collect_submodules(crate: Crate, module_item, prefix):
    """Yield (dotted_path, module_item) for public submodules, recursively."""
    for iid, it in crate.public_items(module_item):
        if crate.kind_of(it) == "module":
            sub = f"{prefix}::{it['name']}"
            yield sub, it
            yield from collect_submodules(crate, it, sub)


def frontmatter(title, description, icon="cube"):
    return [
        "---",
        f'title: "{title}"',
        f'description: "{description}"',
        f'icon: "{icon}"',
        "---",
        "",
    ]


def generated_note(crate: Crate, commit):
    ver = crate.version or "workspace"
    return (
        f"<Note>\nGenerated from rustdoc JSON at `{crate.docsrs} v{ver}` "
        f"(commit `{commit}`). Do not edit by hand; regenerate with "
        f"`python3 docs/scripts/gen_api.py` (see the script header for the "
        f"rustdoc commands). Item links lead to full signatures and method "
        f"docs on docs.rs.\n</Note>\n"
    )


def main():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip() or "unknown"

    OUT.mkdir(parents=True, exist_ok=True)
    nav_pages = []

    for json_name, docsrs_name, path_root, page_stem in CRATES:
        src = DOC_JSON / json_name
        if not src.exists():
            sys.exit(f"missing {src}; run the rustdoc commands in the script header first")
        crate = Crate(json.loads(src.read_text()), docsrs_name, path_root)
        root_item = crate.item(crate.root)

        if page_stem is None:
            # dspy-rs: crate-root page + one page per top-level module
            top_modules = [
                (iid, it)
                for iid, it in crate.public_items(root_item)
                if crate.kind_of(it) == "module"
            ]
            non_modules = [
                (iid, it)
                for iid, it in crate.public_items(root_item)
                if crate.kind_of(it) != "module"
            ]

            lines = frontmatter(
                "dspy_rs (crate root)",
                "Everything importable directly from dspy_rs",
                "box-open",
            )
            lines.append(generated_note(crate, commit))
            lines.append(
                "The crate root re-exports the surface most programs use, so"
                " `use dspy_rs::{Predict, Signature, configure}` works without"
                " module paths. Items are listed here once with their home module"
                " linked; the module pages list everything else.\n"
            )
            lines += render_tables(crate, non_modules)
            lines.append("## Modules")
            lines.append("")
            lines.append("| Module | Description |")
            lines.append("|---|---|")
            for iid, it in sorted(top_modules, key=lambda t: t[1]["name"]):
                doc = first_sentence(it.get("docs", ""))
                lines.append(f"| [`{it['name']}`](/docs/api/{it['name']}) | {doc} |")
            lines.append("")
            (OUT / "dspy-rs.mdx").write_text("\n".join(lines))
            nav_pages.append("docs/api/dspy-rs")

            for iid, it in sorted(top_modules, key=lambda t: t[1]["name"]):
                name = it["name"]
                lines = frontmatter(
                    f"dspy_rs::{name}",
                    first_sentence(it.get("docs", "")) or f"Public API of the {name} module",
                )
                lines.append(generated_note(crate, commit))
                if it.get("docs"):
                    lines.append(it["docs"].strip().split("\n\n")[0] + "\n")
                lines += render_tables(crate, crate.public_items(it))
                for sub_path, sub_item in collect_submodules(crate, it, name):
                    sub_items = crate.public_items(sub_item)
                    sub_items = [
                        (i, x) for i, x in sub_items if crate.kind_of(x) != "module"
                    ]
                    if not sub_items:
                        continue
                    lines.append(f"## `{sub_path}`")
                    lines.append("")
                    if sub_item.get("docs"):
                        lines.append(first_sentence(sub_item["docs"]) + "\n")
                    lines += render_tables(crate, sub_items, heading_level="###")
                (OUT / f"{name}.mdx").write_text("\n".join(lines))
                nav_pages.append(f"docs/api/{name}")
        else:
            # companion crates: one page each
            lines = frontmatter(
                path_root,
                first_sentence(root_item.get("docs", ""))
                or f"Public API of the {docsrs_name} crate",
            )
            lines.append(generated_note(crate, commit))
            if root_item.get("docs"):
                lines.append(root_item["docs"].strip().split("\n\n")[0] + "\n")
            lines += render_tables(crate, crate.public_items(root_item))
            for sub_path, sub_item in collect_submodules(crate, root_item, path_root):
                sub_items = [
                    (i, x)
                    for i, x in crate.public_items(sub_item)
                    if crate.kind_of(x) != "module"
                ]
                if not sub_items:
                    continue
                lines.append(f"## `{sub_path}`")
                lines.append("")
                lines += render_tables(crate, sub_items, heading_level="###")
            (OUT / f"{page_stem}.mdx").write_text("\n".join(lines))
            nav_pages.append(f"docs/api/{page_stem}")

    print("generated pages:")
    for p in nav_pages:
        print(" ", p)


if __name__ == "__main__":
    main()
