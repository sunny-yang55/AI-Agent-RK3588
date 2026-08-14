import json
from pathlib import Path


class EntityResolver:

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        config_file = root / "config" / "entity_alias.json"

        if not config_file.exists():

            raise FileNotFoundError(f"Missing {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:

            self.entities = json.load(f)

        # ==========================
        # Build Alias Index
        # ==========================

        self.alias_index = {}

        self.entity_count = 0

        for category, items in self.entities.items():

            for entity, info in items.items():

                self.entity_count += 1

                standard_name = entity

                aliases = info.get("aliases", [])

                # 标准名称也加入索引

                self.alias_index[standard_name] = standard_name

                for alias in aliases:

                    self.alias_index[alias] = standard_name

        # print("[EntityResolver V2]")

        # print(f"Entities : {self.entity_count}")

        # print(f"Aliases  : {len(self.alias_index)}")

        # print("Index Ready")

    def extract_entity(self, text):

        if not text:
            return None

        candidates = sorted(self.alias_index.keys(), key=len, reverse=True)

        for alias in candidates:

            if alias in text:

                return self.alias_index[alias]

        return None

    def resolve(self, text: str):

        if not text:

            return text, None

        result = text

        matched_entity = None

        # 长词优先匹配

        candidates = sorted(self.alias_index.keys(), key=len, reverse=True)

        for alias in candidates:

            if alias in result:

                target = self.alias_index[alias]

                if alias != target:

                    # print(f"[EntityResolver] {alias} -> {target}")

                    result = result.replace(alias, target)

                # 保存最后一次命中的标准实体

                matched_entity = target

                break

        return result, matched_entity
