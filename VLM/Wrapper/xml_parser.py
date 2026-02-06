import re
from typing import Dict, Optional

class XMLParser:
    """
    Parses XML-formatted actions from LLM output.
    Design strictly follows Prime Intellect verifiers.XMLParser pattern.
    """
    def __init__(self, fields: Dict[str, str]):
        """
        Args:
            fields: A dict mapping field names to their expected tag names.
                    e.g. {"answer": "action"}
        """
        self.fields = fields
        # In this specific case, we care about the 'answer' field which maps to <action>
        self.tag_name = fields.get("answer", "action")

    def parse(self, text: str) -> Optional[str]:
        """
        Extracts the content of the action tag.
        Returns None if not found, malformed, or if multiple tags exist.
        """
        pattern = f"<{self.tag_name}>(.*?)</{self.tag_name}>"
        # Find ALL matches to enforce single-action constraint
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if len(matches) == 1:
            return matches[0].strip()
        elif len(matches) > 1:
            # Reject if agent output multiple actions (hallucination)
            return None
        return None

    def format_reward(self, text: str) -> float:
        """
        Returns 1.0 if the format is correct (tag exists), 0.0 otherwise.
        """
        return 1.0 if self.parse(text) is not None else 0.0
