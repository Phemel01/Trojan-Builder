"""Ground routes on the arena supplied in hello, built once per match."""
from array import array
import heapq
import json
import math
from pathlib import Path
import sys


def direction(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > 0.01 else (0.0, 0.0)


class FloorMap:
    step = 2.5

    def __init__(self, hello):
        self.mask = None
        self.heights = None
        root = Path(hello.get("map", {}).get("dir", ".")) / "v2"
        try:
            world = json.loads((root / "world.json").read_text())
        except (OSError, ValueError):
            return
        solids = world.get("prisms", [])
        bounds = world.get("bounds")
        if not bounds and not solids:
            return
        bounds = bounds or dict(minX=-620, maxX=620, minY=-620, maxY=620)
        s = self.step
        self.x0 = math.floor(bounds["minX"] / s) * s
        self.y0 = math.floor(bounds["minY"] / s) * s
        self.w = math.ceil((bounds["maxX"] - self.x0) / s) + 1
        self.h = math.ceil((bounds["maxY"] - self.y0) / s) + 1
        w, h = self.w, self.h
        blocked = bytearray(w * h)
        cfg = hello.get("config", {}).get("mech", {})
        self._read_heights(root, world.get("terrain"), blocked, cfg.get("max_slope_deg", 50))
        self.body_height = cfg.get("height", 9)
        self.ledge = cfg.get("step_up", 1.2)
        for part in solids:
            self._paint(part, blocked)
        self.mask = blocked[:]
        # Leave a footprint margin, including along thin walls that miss cell centers.
        for i, solid in enumerate(blocked):
            if solid and w <= i < w * (h - 1):
                self.mask[i - 1] = self.mask[i + 1] = self.mask[i - w] = self.mask[i + w] = 1
        for y in range(h):
            self.mask[y * w] = self.mask[y * w + w - 1] = 1
        self.mask[:w] = self.mask[-w:] = bytes([1]) * w
        # A courtyard, ridge or closed building can partition the walkable floor.
        # Only choose destinations in the same connected component as the avatar.
        self.components = array("I", [0])*len(self.mask)
        islands = 0
        for start in range(len(self.mask)):
            if self.mask[start] or self.components[start]:
                continue
            islands += 1
            self.components[start] = islands
            queue = [start]
            for node in queue:
                for other in (node-1, node+1, node-w, node+w):
                    if not self.mask[other] and not self.components[other]:
                        self.components[other] = islands
                        queue.append(other)

    def _read_heights(self, root, spec, blocked, max_slope):
        if not spec:
            return
        try:
            raw = array("h")
            raw.frombytes((root / "terrain.i16").read_bytes())
        except (OSError, ValueError):
            return
        nx, ny = spec["nx"], spec["ny"]
        if len(raw) != nx * ny:
            return
        if sys.byteorder != "little":
            raw.byteswap()
        self.heights = array("h", [0]) * len(blocked)
        scale = spec.get("scale", 0.01)
        self.zscale = scale
        raw_step = spec["step"]
        grad_limit = (2 * raw_step * math.tan(math.radians(max_slope)) / scale) ** 2
        xi = [min(nx - 2, max(1, round((self.x0 + x*self.step - spec["minX"]) / raw_step)))
              for x in range(self.w)]
        for y in range(self.h):
            ry = min(ny - 2, max(1, round((self.y0 + y*self.step - spec["minY"]) / raw_step)))
            row = y * self.w
            for x, rx in enumerate(xi):
                j = ry * nx + rx
                self.heights[row+x] = raw[j]
                gx = raw[j+1] - raw[j-1]
                gy = raw[j+nx] - raw[j-nx]
                if gx*gx + gy*gy > grad_limit:
                    blocked[row+x] = 1

    def _paint(self, part, blocked):
        rings = [list(zip(flat[::2], flat[1::2])) for flat in [part["ring"], *part.get("holes", [])] if flat]
        if not rings:
            return
        s, w = self.step, self.w
        base, top = part["base"], part["top"]
        def mark(col, row):
            if 0 <= col < w and 0 <= row < self.h:
                i = row * w + col
                z = self.heights[i] * self.zscale if self.heights is not None else 0.0
                if top > z + self.ledge and base < z + self.body_height:
                    blocked[i] = 1
        # Even/odd scanline fill respects courtyard holes. Trace ring edges separately.
        spans = {}
        for ring in rings:
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
                steps = max(1, math.ceil(math.hypot(x2-x1, y2-y1) / (0.5*s)))
                for k in range(steps):
                    f = k/steps
                    mark(round((x1+(x2-x1)*f-self.x0)/s), round((y1+(y2-y1)*f-self.y0)/s))
                if y1 == y2:
                    continue
                lo = max(0, math.ceil((min(y1, y2)-self.y0)/s))
                hi = min(self.h, math.ceil((max(y1, y2)-self.y0)/s))
                for row in range(lo, hi):
                    y = self.y0 + row*s
                    spans.setdefault(row, []).append(x1 + (x2-x1)*(y-y1)/(y2-y1))
        for row, xs in spans.items():
            xs.sort()
            for left, right in zip(xs[::2], xs[1::2]):
                for col in range(max(0, math.ceil((left-self.x0)/s)), min(w, math.floor((right-self.x0)/s)+1)):
                    mark(col, row)

    def cell(self, point):
        x = min(self.w-1, max(0, round((point[0]-self.x0)/self.step)))
        y = min(self.h-1, max(0, round((point[1]-self.y0)/self.step)))
        return y*self.w + x

    def xy(self, i):
        return (self.x0+(i % self.w)*self.step, self.y0+(i // self.w)*self.step)

    def ground(self, point):
        if self.heights is not None:
            return self.heights[self.cell(point)] * self.zscale
        return None

    def nearest(self, i, component=None, max_radius=32):
        if not self.mask[i] and (component is None or self.components[i] == component):
            return i
        cx, cy = i % self.w, i // self.w
        for r in range(1, max_radius+1):
            choices = []
            for y in range(max(1, cy-r), min(self.h-1, cy+r+1)):
                for x in ((cx-r,cx+r) if abs(y-cy)<r else range(cx-r,cx+r+1)):
                    if (0 < x < self.w-1 and not self.mask[y*self.w+x]
                            and (component is None or self.components[y*self.w+x] == component)):
                        choices.append(((x-cx)**2+(y-cy)**2, y*self.w+x))
            if choices:
                return min(choices)[1]
        return None

    def clear(self, a, b):
        """Visit grid cells on the segment, including both sides of diagonal corners."""
        if not self.mask:
            return True
        w, mask = self.w, self.mask
        i, end = self.cell(a), self.cell(b)
        if mask[i] or mask[end]:
            return False
        x0, y0, x1, y1 = i%w, i//w, end%w, end//w
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = (1 if x1>x0 else -1), (w if y1>y0 else -w)
        decision = dy-dx
        while i != end:
            if decision == 0:
                if mask[i+sx] or mask[i+sy]:
                    return False
                i += sx+sy
                decision += 2*(dy-dx)
            elif decision < 0:
                i += sx
                decision += 2*dy
            else:
                i += sy
                decision -= 2*dx
            if mask[i]:
                return False
        return True


class Navigator:
    def __init__(self, floor, side):
        self.floor, self.side = floor, side
        self.goal = None
        self.component = None
        self.needs_jet = False
        self.search = None
        self.path = []
        self.retry_tick = 0
        self.failed = False

    def local(self, pos, desired, look=7):
        f = self.floor
        if not f.mask or not any(desired):
            return desired
        if f.mask[f.cell(pos)]:
            cell = f.nearest(f.cell(pos))
            return direction(pos, f.xy(cell)) if cell is not None else desired
        for degrees in (0, 32, -32, 58, -58, 85, -85, 115, -115, 155, -155, 180):
            yaw = math.atan2(desired[1], desired[0]) + self.side * math.radians(degrees)
            d = math.cos(yaw), math.sin(yaw)
            if f.clear(pos, (pos[0]+look*d[0], pos[1]+look*d[1])):
                return d
        return desired

    def _advance(self, dst):
        f = self.floor
        w, mask = f.w, f.mask
        queue, parent, costs = self.search
        tx, ty = dst % w, dst // w
        # Bound the work on each decision. A long detour continues on subsequent ticks.
        for _ in range(650):
            if not queue:
                self.search = None
                self.failed = True
                return
            _, cost, i = heapq.heappop(queue)
            if i == dst:
                path = []
                while parent[i] is not None:
                    path.append(i)
                    i = parent[i]
                self.path = list(reversed(path))
                self.search = None
                return
            if cost != costs[i]:
                continue
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                j = i+dx+dy*w
                if mask[j] or (dx and dy and (mask[i+dx] or mask[i+dy*w])):
                    continue
                c = cost + (14 if dx and dy else 10)
                if c >= costs.get(j, math.inf):
                    continue
                hx, hy = abs(j%w-tx), abs(j//w-ty)
                h = 10*max(hx,hy) + 4*min(hx,hy)
                costs[j], parent[j] = c, i
                heapq.heappush(queue, (c+1.18*h, c, j))

    def toward(self, pos, goal, tick):
        f = self.floor
        if not f.mask:
            return direction(pos, goal)
        i = f.cell(pos)
        if f.mask[i]:
            return direction(pos, goal) if self.needs_jet else self.local(pos, direction(pos, goal))
        component = f.components[i]
        if self.goal is None or math.dist(goal, self.goal) > 2.0 or self.component != component:
            self.component, self.goal = component, goal
            self.dst = f.nearest(f.cell(goal), component)
            self.search, self.path, self.failed = None, [], False
            self.retry_tick = tick
        self.needs_jet = self.dst is None
        if self.needs_jet:
            return direction(pos, goal)
        endpoint = goal if f.cell(goal) == self.dst else f.xy(self.dst)
        if math.dist(endpoint,goal)>11 and math.dist(pos,endpoint)<7:
            self.needs_jet = True
            return direction(pos,goal)
        if f.clear(pos, endpoint):
            self.path, self.search, self.failed = [], None, False
            return direction(pos, endpoint)
        if self.path and not f.clear(pos, f.xy(self.path[0])):
            self.path = []
            self.retry_tick = tick
        if not self.path and self.search is None and tick >= self.retry_tick:
            self.search = ([(0,0,i)], {i:None}, {i:0})
            self.retry_tick = tick + 80
            self.failed = False
        if self.search is not None:
            self._advance(self.dst)
            if self.search is not None:
                return (0.0, 0.0)
        if self.path:
            # Skip intermediate cells only when the full connecting segment is free.
            limit = min(24, len(self.path)-1)
            while limit and not f.clear(pos, f.xy(self.path[limit])):
                limit -= 1
            if limit:
                del self.path[:limit]
            waypoint = f.xy(self.path[0])
            if math.dist(pos, waypoint) < f.step*0.65:
                self.path.pop(0)
                if self.path:
                    waypoint = f.xy(self.path[0])
            return direction(pos, waypoint)
        return self.local(pos, direction(pos, endpoint))
