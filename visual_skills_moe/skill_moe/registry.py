from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from .base import SkillMetadata
from .skill_fs import parse_skill


class SkillRegistry:
    """
    Filesystem-based registry: each skill is a directory containing SKILL.md.
    Mirrors Claude/SkillsBench layout.
    """

    def __init__(self, root: str = "skills") -> None:
        self.root = os.path.abspath(root)
        self._skills: Dict[str, SkillMetadata] = {}
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        if not os.path.isdir(self.root):
            return
        for entry in os.listdir(self.root):
            skill_dir = os.path.join(self.root, entry)
            if not os.path.isdir(skill_dir):
                continue
            skill_md = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            meta = parse_skill(skill_md)
            if meta:
                self._skills[meta.name] = meta

    def get(self, name: str) -> Optional[SkillMetadata]:
        return self._skills.get(name)

    def list(self) -> List[str]:
        return sorted(self._skills.keys())

    def __iter__(self) -> Iterator[SkillMetadata]:
        return iter(self._skills.values())
