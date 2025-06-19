import inspect
import spectralmatch
import os
from collections import defaultdict
from mkdocs_gen_files import open as gen_open


def generate_commands_section(module) -> str:
    func_groups = defaultdict(list)

    for name in getattr(module, "__all__", []):
        func = getattr(module, name, None)
        if callable(func):
            doc = inspect.getdoc(func)
            if not doc:
                raise ValueError(f"Missing docstring for function: {name}")

            mod = inspect.getmodule(func)
            if mod and mod.__name__.startswith("spectralmatch."):
                group_key = mod.__name__.split(".")[1]
            elif mod and mod.__name__ == "spectralmatch":
                group_key = os.path.splitext(os.path.basename(inspect.getfile(func)))[0]
            else:
                group_key = mod.__name__ if mod else "unknown"

            func_groups[group_key].append((name, doc.splitlines()[0]))

    # Build commands section
    lines = [""]
    for group in func_groups:
        group_title = group.replace("_", " ").capitalize()
        lines.append(f"### {group_title}\n")
        for name, first_line in func_groups[group]:
            lines.append(f"#### `{name}`\n{first_line.strip()}\n")

    return "\n".join(lines)


def main():
    template_path = os.path.join("docs", "cli.md")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    if "{commands_content}" not in template:
        raise ValueError("Placeholder {commands_content} not found in cli.md")

    commands = generate_commands_section(spectralmatch)
    return template.replace("{commands_content}", commands)


# Write output during mkdocs build
with gen_open("cli.md", "w") as f:
    f.write(main())