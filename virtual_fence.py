"""
ai/virtual_fence.py

Implements Module F (Section 8) -- Virtual Fence Intrusion Detection.

Workflow per Section 8.2: Detect -> Track -> Check Position -> Compare
With Virtual Boundary -> Boundary Crossed? -> Alert / Continue.

The spec defines the concept and workflow, not a specific geometry
representation, so a straight horizontal/vertical/diagonal line (defined
by two points) is used here -- the simplest thing that satisfies
"a defined boundary inside the surveillance area" (Section 8.1).
"""


def _side_of_line(point, line_p1, line_p2):
    """Returns >0, <0, or 0 depending which side of the line the point is on."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


class VirtualFence:
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
        self.last_side = {}  # track_id -> last side sign

    def check(self, track_id, centroid):
        """
        Returns True the frame an intrusion (boundary crossing) is detected
        for this track_id, False otherwise.
        """
        side = _side_of_line(centroid, self.p1, self.p2)
        side_sign = 1 if side > 0 else (-1 if side < 0 else 0)

        crossed = False
        if track_id in self.last_side and self.last_side[track_id] != 0 and side_sign != 0:
            if self.last_side[track_id] != side_sign:
                crossed = True

        self.last_side[track_id] = side_sign
        return crossed
