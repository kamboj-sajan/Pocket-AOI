# System

You read reference designators printed on PCB silkscreen. Answer with one JSON object and nothing else.

# User

The image shows one component outlined in red and the board around it.
Read the reference designator printed for the outlined part, for example R3, C12, U1, D2, Q1 or LED1.
Text may be rotated, small or partly hidden by the part. If several labels are visible, pick the one closest to the outline.
Detected class: {part_class}. Valid references on this board: {candidate_refs}.
If you cannot read it, return null for ref. Never guess.
Return exactly:
{"ref": "R3", "alternatives": [], "confidence": 0.0}
