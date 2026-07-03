"""
Post-processing utilities for entity extraction pipeline.
Handles position correction when the LLM returns slightly wrong character offsets.
"""


def fix_position(source: str, entity_text: str, position: tuple[int, int]) -> list[int]:
    """
    Correct the character-level position of an entity in the source text.
    
    The LLM may return positions that are slightly off. This function:
    1. Checks if source[start:end] == entity_text  →  if yes, return as-is.
    2. Searches a local window around the reported position for an exact match.
    3. Falls back to the first occurrence via str.find().
    4. Returns the original position if the text is not found at all.
    
    Args:
        source: The full clinical text.
        entity_text: The exact entity string reported by the LLM.
        position: (start, end) tuple from the LLM.
        
    Returns:
        [start, end] list with corrected positions.
    """
    start, end = position
    
    # 1. Exact match at reported position
    if source[start:end] == entity_text:
        return [start, end]
    
    # 2. Search within a local window (±50 chars) around the reported start
    window = 50
    search_start = max(0, start - window)
    search_end = min(len(source), end + window)
    local_idx = source[search_start:search_end].find(entity_text)
    if local_idx != -1:
        corrected_start = search_start + local_idx
        return [corrected_start, corrected_start + len(entity_text)]
    
    # 3. Fallback: find first occurrence in entire source
    global_idx = source.find(entity_text)
    if global_idx != -1:
        return [global_idx, global_idx + len(entity_text)]
    
    # 4. Text not found — return original position unchanged
    return [start, end]
