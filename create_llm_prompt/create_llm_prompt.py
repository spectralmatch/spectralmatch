import os
import fnmatch
import ast
from html import escape
from mkdocs_gen_files import open as gen_open


def create_initial_prompt_text() -> str:
    return (
        "# LLM Prompt\n\n"
        "The following content includes function signatures and docstrings from Python source files, as well as relevant Markdown documentation. Each section is labeled by its relative file path. Use this as context to understand the project structure, purpose, and functionality.\n\n"
    )


def parse_python_files_to_prompt_text(input_directory="spectralmatch", include_filter="*.py", only_include_function_headers=True) -> str:
    prompt_lines = []
    for root, _, files in os.walk(input_directory):
        for filename in fnmatch.filter(files, include_filter):
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, input_directory)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                tree = ast.parse(code)
            except SyntaxError:
                continue

            lines_for_file = []
            if only_include_function_headers:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        args = [arg.arg for arg in node.args.args]
                        sig = f"def {node.name}({', '.join(args)}):"
                        doc = ast.get_docstring(node)
                        if doc:
                            sig += f'\n    """{doc}"""'
                        lines_for_file.append(sig)
            else:
                lines_for_file.append(code)

            if lines_for_file:
                prompt_lines.append(f"### File: {rel_path}")
                prompt_lines.extend(lines_for_file)

    if not prompt_lines:
        return ""
    return "## Python Section\n" + "\n\n".join(prompt_lines) + "\n"


def parse_markdown_files_to_prompt_text(input_directory="docs", include_filter="*.md", exclude_filter="*prompt*") -> str:
    prompt_lines = []
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if fnmatch.fnmatch(filename, include_filter):
                if exclude_filter and fnmatch.fnmatch(filename, exclude_filter):
                    continue
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, input_directory)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                prompt_lines.append(f"### File: {rel_path}\n{content.strip()}\n")
    if not prompt_lines:
        return ""
    return "## Markdown Section\n" + "\n\n".join(prompt_lines) + "\n"


def main() -> str:
    # Build prompt content
    parts = [
        create_initial_prompt_text(),
        parse_python_files_to_prompt_text(),
        parse_markdown_files_to_prompt_text()
    ]
    prompt = "\n".join(filter(None, parts))

    # Load template and inject
    template_path = os.path.join("docs", "llm_prompt.md")
    with open(template_path, "r", encoding="utf-8") as tf:
        template = tf.read()

    return template.replace("{prompt_content}", prompt)


# Write to output during mkdocs build
with gen_open("llm_prompt.md", "w") as f:
    f.write(main())