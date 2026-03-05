from __future__ import annotations

import os
import re
from typing import Optional

from .base import SkillMetadata


def parse_skill(skill_md_path: str) -> Optional[SkillMetadata]:
    """
    Parse YAML-like front matter from SKILL.md.
    Supports the minimal fields used by SkillsBench examples: name, description, tags, when_to_use.
    """
    if not os.path.isfile(skill_md_path):
        return None

    with open(skill_md_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.match(r"---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return None
    front_matter, _body = match.groups()

    data = {}
    for line in front_matter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')

    name = data.get("name")
    description = data.get("description", "")
    tags = [t.strip() for t in data.get("tags", "").split(",") if t.strip()]
    when_to_use = data.get("when_to_use")

    if not name:
        return None

    return SkillMetadata(
        name=name,
        description=description,
        tags=tags,
        when_to_use=when_to_use,
        path=os.path.abspath(os.path.dirname(skill_md_path)),
    )
