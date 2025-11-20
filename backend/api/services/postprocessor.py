from typing import Optional, Dict, Any, Union
import re

class Postprocessor:
    """
    A utility class for extracting structured information from the raw output
    of a Vision-Language Model (VLM). The postprocessor attempts to identify:

    - A classification flag (boolean, label, or `*Flag: value` line)
    - A textual explanation, if present
    - The original raw string

    This class is robust to multiple formatting styles typically returned by
    generative models and provides a clean interface for downstream usage.
    """
    def __init__(self):
        """
        Initialize the Postprocessor instance.

        The class does not require configuration at instantiation, but is
        structured this way for future extensibility (e.g., custom parsing rules).
        """
        pass

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse the raw model output into structured fields.

        Args:
            raw_response (str):
                The direct, unprocessed text returned by the VLM.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "flag": The extracted flag (boolean, string label, or `Flag: value` line)
                - "explanation": A textual explanation, if detected
                - "raw": The original unmodified response
        """
        flag = self._extract_flag(raw_response)
        explanation = self._extract_explanation(raw_response)

        return {
            "flag": flag,
            "explanation": explanation,
            "raw": raw_response,
        }

    def _extract_flag(self, text: str) -> Optional[Union[bool, str]]:
        """
        Extract the classification flag from the VLM output.

        Priority order:
        1. If the model returns a full line containing a flag, such as:
           - "CleanFlag: true"
           - "ResultFlag = dirty"
           the entire line is returned.
        2. If the response is exactly a boolean ("true"/"false"), return a Python bool.
        3. If the response is a single word token (e.g., "clean", "dirty"), return it as-is.
        4. Otherwise, return None.

        Args:
            text (str): Raw VLM output.

        Returns:
            Optional[Union[bool, str]]:
                A parsed flag following the above priority rules.
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
        Extract an explanation string from the model output.

        Looks for common explanation formats such as:
        - "Explanation: ..."
        - "Explanation = ..."

        Everything after the delimiter is returned.

        Args:
            text (str): Raw VLM output.

        Returns:
            Optional[str]: The explanation if found, otherwise None.
        """
        parts = re.split(r"Explanation\s*[:=]\s*", text, flags=re.IGNORECASE)

        if len(parts) <= 1:
            return None

        explanation = parts[-1].strip()
        
        return explanation if explanation else None