from typing import Optional, Dict, Any, Union
import re

class Postprocessor:
    """
    Postprocess raw VLM output to extract a flag that may be:
    - Full line: "<SomethingFlag>: value"
    - Just a boolean
    - Just a label (clean/dirty)
    """

    def __init__(self):
        pass

    def parse(self, raw_response: str) -> Dict[str, Any]:
        flag = self._extract_flag(raw_response)
        explanation = self._extract_explanation(raw_response)

        return {
            "flag": flag,
            "explanation": explanation,
            "raw": raw_response,
        }

    def _extract_flag(self, text: str) -> Optional[Union[bool, str]]:
        """
        Priority:
        1. Any line matching "*Flag: value" or "*Flag = value" → return full line
        2. Pure boolean
        3. Single word class label
        4. None
        """
        full_line_pattern = r"([A-Za-z0-9_]*Flag\s*[:=]\s*.+)"
        matches = re.findall(full_line_pattern, text, flags=re.IGNORECASE)

        if matches:
            return matches[-1].strip()

        cleaned = text.strip()

        if cleaned.lower() == "true":
            return True
        if cleaned.lower() == "false":
            return False

        tokens = cleaned.split()
        if len(tokens) == 1:
            return cleaned

        return None

    def _extract_explanation(self, text: str) -> Optional[str]:
        """
        Extract explanation after any format:
        "Explanation:" or "Explanation ="
        """
        parts = re.split(r"Explanation\s*[:=]\s*", text, flags=re.IGNORECASE)

        if len(parts) <= 1:
            return None

        explanation = parts[-1].strip()
        return explanation if explanation else None