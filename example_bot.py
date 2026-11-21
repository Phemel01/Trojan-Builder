"""Trojan Ghost: survive by not being seen.

Placement is the order mechs die in, so the bot plays for time, not kills:
  * never shoots first (a shot is heard 320 m away and turns a mech that can't see you into one
    that hunts you; dealing damage also blocks hp regen for regen_delay s);
  * hides in low-exposure spots inside the next zone circle, chosen from a map grid precomputed
    offline from the official arena (buildings and dense canopy block sight; see tools/);
  * walks there on an A* path through hidden alleys, time-sliced so no decision exceeds the budget;
  * when hit or seen up close, faces the attacker (front armor) and moves to the nearest cell whose
    line of sight to every threat is blocked, firing back only while it is actually being hit;
  * once hidden, stands still (silent) so regen (45 hp/s after 5 s) refills it;
  * in the last circles, where hiding is impossible, fights with lead-aimed gun (and laser if held).
"""
import heapq
import json
import math
import os
import time

import mechfight as mf

TAU = 2 * math.pi
SQ2 = math.sqrt(2.0)
NB8 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
UPGRADE_PREF = ["nanites", "plating", "armor", "sensors", "servos", "gyros", "capacitor",
                "reactor", "jets", "gun_damage", "gun_rate", "gun_mag", "gun_velocity"]


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)


class Grid:
    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = json.loads(f.readline())
            self.h = hdr
            n4, n2 = hdr["nx4"] * hdr["ny4"], hdr["nx2"] * hdr["ny2"]
            self.walk = f.read(n4)
            self.expo = f.read(n4)
            self.vis = f.read(n2)
            g = f.read(2 * n2)
            self.clr = f.read(n4)
        self.gnd = memoryview(g).cast("h")
        self.minX, self.minY = hdr["minX"], hdr["minY"]
        self.nx4, self.ny4, self.nx2, self.ny2 = hdr["nx4"], hdr["ny4"], hdr["nx2"], hdr["ny2"]

    # 4 m cells
    def cell(self, x, y):
        return int((x - self.minX) / 4.0), int((y - self.minY) / 4.0)

    def center(self, i, j):
        return self.minX + (i + 0.5) * 4.0, self.minY + (j + 0.5) * 4.0

    def ok(self, i, j):
        return 0 <= i < self.nx4 and 0 <= j < self.ny4 and self.walk[j * self.nx4 + i] == 1

    def exposure(self, i, j):
        return self.expo[j * self.nx4 + i] / 255.0

    def snap(self, i, j, rmax=6):
        if self.ok(i, j):
            return i, j
        for r in range(1, rmax + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    if self.ok(i + di, j + dj):
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, i + di, j + dj)
            if best:
                return best[1], best[2]
        return None

    # 2 m sight grid
    def ground(self, x, y):
        i, j = int((x - self.minX) / 2.0), int((y - self.minY) / 2.0)
        if 0 <= i < self.nx2 and 0 <= j < self.ny2:
            return self.gnd[j * self.nx2 + i] / 100.0
        return 0.0

    def los(self, ax, ay, az, bx, by, bz):
        """Eye (az+8) to body centre (bz+4.5): blocked by a sight-blocking building cell or terrain."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        n = int(d / 1.6)
        if n < 2:
            return True
        ea, eb = az + 8.0, bz + 4.5
        mx, my, nx2 = self.minX, self.minY, self.nx2
        vis, gnd = self.vis, self.gnd
        for k in range(1, n):
            f = k / n
            x, y = ax + dx * f, ay + dy * f
            i, j = int((x - mx) * 0.5), int((y - my) * 0.5)
            if not (0 <= i < nx2 and 0 <= j < self.ny2):
                return False
            idx = j * nx2 + i
            if vis[idx]:
                return False
            if gnd[idx] / 100.0 > ea + (eb - ea) * f:
                return False
        return True

    def walk_line(self, ax, ay, bx, by):
        """Straight segment stays on walkable 4 m cells."""
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d / 2.0))
        for k in range(1, n + 1):
            f = k / n
            i, j = self.cell(ax + (bx - ax) * f, ay + (by - ay) * f)
            if not self.ok(i, j):
                return False
        return True


class AStar:
    """Time-sliced A* on the 4 m grid; cost favours hidden cells."""

    def __init__(self, g, start, goal, expo_w=2.0):
        self.g, self.goal, self.w = g, goal, expo_w
        self.open = [(0.0, 0.0, start)]
        self.came = {start: None}
        self.cost = {start: 0.0}
        self.done = False
        self.path = None

    def h(self, c):
        dx, dy = abs(c[0] - self.goal[0]), abs(c[1] - self.goal[1])
        return (dx + dy) + (SQ2 - 2) * min(dx, dy)

    def step(self, deadline):
        g = self.g
        walk, expo, nx, ny = g.walk, g.expo, g.nx4, g.ny4
        op, came, cost, goal, w = self.open, self.came, self.cost, self.goal, self.w
        n = 0
        while op:
            n += 1
            if (n & 255) == 0 and time.perf_counter() > deadline:
                return False
            f, c0, cur = heapq.heappop(op)
            if cur == goal:
                p = []
                while cur is not None:
                    p.append(cur)
                    cur = came[cur]
                self.path = p[::-1]
                self.done = True
                return True
            if c0 > cost.get(cur, 1e18):
                continue
            ci, cj = cur
            for di, dj, dl in NB8:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                idx = nj * nx + ni
                if walk[idx] != 1:
                    continue
                if di and dj and (walk[cj * nx + ni] != 1 or walk[nj * nx + ci] != 1):
                    continue
                nc = c0 + dl * (1.0 + w * expo[idx] / 255.0)
                nb = (ni, nj)
                if nc < cost.get(nb, 1e18):
                    cost[nb] = nc
                    came[nb] = cur
                    heapq.heappush(op, (nc + self.h(nb), nc, nb))
        self.done = True
        return True


class Bot:
    name = "trojan-ghost"
    BUDGET_S = 0.015   # compute slice per decision (the wire budget is 50 ms; margin for a throttled CPU)
    DIST_W = float(os.environ.get("GHOST_DIST_W", "1.5"))   # hideout: exposure vs travel distance trade-off

    def __init__(self, hello):
        cfg = hello.get("config", {}) or {}
        self.cfg = cfg
        mech = cfg.get("mech", {}) or {}
        self.walk_speed = float(mech.get("walk_speed", 11.7))
        self.regen_delay = float(mech.get("regen_delay", 5.0))
        vision = cfg.get("vision", {}) or {}
        self.near_range = float(vision.get("near_range", 45.0))
        self.vis_range = float(vision.get("range", 200.0))
        self.fov = math.radians(float(vision.get("fov_deg", 120.0)))
        w = cfg.get("weapons", {}) or {}
        self.gun_speed = float((w.get("gun") or {}).get("speed", 252.0))
        self.gun_range = float((w.get("gun") or {}).get("range", 230.0))
        self.laser_range = float((w.get("laser") or {}).get("range", 280.0))
        self.laser_charge = float((w.get("laser") or {}).get("charge", 0.75))
        self.reach = float((cfg.get("loot") or {}).get("pickup_radius", 4.0))
        self.g = None
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            self.g = Grid(os.path.join(here, "arena_grid.bin"))
        except Exception as e:  # keep playing without the map
            mf.log(f"ghost: no grid ({e}); falling back to direct movement")
        self.zone_key = None
        self.hide = None          # (x, y) hideout centre
        self.hide_aim = 0.0
        self.planner = None
        self.path = None
        self.path_i = 0
        self.pending_goal = None
        self.attackers = {}       # id -> (t, est_x, est_y)
        self.last_hit_t = -99.0
        self.last_dealt_t = -99.0
        self.known = {}           # mech id -> (t, x, y, z, vx, vy)
        self.npcs = {}
        self.sounds = []          # (t, bearing, kind)
        self.mode = "START"
        self.cover = None
        self.cover_t = -99.0
        self.stuck_ref = None
        self.stuck_until = -1.0
        self.stuck_side = 1.0
        self.scan_t = 0.0
        self.scan_k = 0
        self.laser_hold = False
        self.bad_hides = set()
        self.no_path_t = -99.0
        self.no_path_n = 0
        self.stats = {"modes": {}, "dmg": {}, "shots": 0}

    # ------------------------------------------------------------------ helpers
    def _deadline(self):
        return self.t0 + self.BUDGET_S

    def _in_next(self, x, y, zone, margin=0.0):
        nx, ny = zone["next_center"]
        return math.hypot(x - nx, y - ny) <= zone["next_radius"] - margin

    def _choose_hide(self, x, y, zone, threats, extra_avoid=()):
        """Pick a low-exposure walkable cell inside the next circle, near us, not in threats' sight."""
        g = self.g
        nx_, ny_ = zone["next_center"]
        r = zone["next_radius"]
        margin = max(10.0, 0.12 * r)
        rr = r - margin
        if rr < 6:
            return None
        ci, cj = g.cell(nx_, ny_)
        R = int(rr / 4.0) + 1
        stride = 1 if R < 40 else 2
        top = []                     # five best (score, i, j)
        drops = extra_avoid
        for dj in range(-R, R + 1, stride):
            for di in range(-R, R + 1, stride):
                i, j = ci + di, cj + dj
                if not g.ok(i, j) or (i, j) in self.bad_hides:
                    continue
                cx, cy = g.center(i, j)
                dc = math.hypot(cx - nx_, cy - ny_)
                if dc > rr:
                    continue
                e = g.exposure(i, j)
                dme = math.hypot(cx - x, cy - y)
                s = e * 1.0 + self.DIST_W * dme / 300.0 + 0.25 * (dc / max(r, 1.0))
                if g.clr[j * g.nx4 + i] < 50:
                    s += 0.05          # tight spots are harder to leave
                for (px, py) in drops:
                    if math.hypot(cx - px, cy - py) < 45:
                        s += 0.4
                if len(top) < 5:
                    top.append((s, i, j))
                    top.sort()
                elif s < top[-1][0]:
                    top[-1] = (s, i, j)
                    top.sort()
        for s, i, j in top:
            hx, hy = g.center(i, j)
            hz = g.ground(hx, hy)
            seen = False
            for (tx, ty, tz) in threats:
                if math.hypot(tx - hx, ty - hy) < self.vis_range and g.los(tx, ty, tz, hx, hy, hz):
                    seen = True
                    break
            if not seen:
                return (i, j)
            self.bad_hides.add((i, j))
        return (top[0][1], top[0][2]) if top else None

    def _open_dir(self, x, y, z):
        """Yaw of the longest open sight line from (x, y) (one march per direction over the sight grid)."""
        g = self.g
        best, by = -1.0, 0.0
        mx, my, nx2, ny2, vis = g.minX, g.minY, g.nx2, g.ny2, g.vis
        for k in range(16):
            a = TAU * k / 16
            ca, sa = math.cos(a), math.sin(a)
            d = 0.0
            while d < 200.0:
                d += 2.0
                i, j = int((x + ca * d - mx) * 0.5), int((y + sa * d - my) * 0.5)
                if not (0 <= i < nx2 and 0 <= j < ny2) or vis[j * nx2 + i]:
                    break
            if d > best:
                best, by = d, a
        return by

    def _cover_from(self, x, y, z, threats, zone=None):
        """Nearest walkable cell within ~100 m, inside the zone, whose sight line to every threat is blocked."""
        g = self.g
        zc, zr = (zone["center"], zone["radius"] - 6.0) if zone else ((x, y), 1e9)
        ci, cj = g.cell(x, y)
        best = None
        for dj in range(-14, 15, 2):
            for di in range(-14, 15, 2):
                i, j = ci + di, cj + dj
                if not g.ok(i, j):
                    continue
                cx, cy = g.center(i, j)
                d = math.hypot(cx - x, cy - y)
                if d < 4 or d > 56 or math.hypot(cx - zc[0], cy - zc[1]) > zr:
                    continue
                if best is not None and d >= best[0]:
                    continue
                cz = g.ground(cx, cy)
                hidden = True
                for (tx, ty, tz) in threats:
                    if g.los(tx, ty, tz, cx, cy, cz):
                        hidden = False
                        break
                if hidden and g.walk_line(x, y, cx, cy):
                    best = (d, cx, cy)
                if time.perf_counter() > self._deadline() + 0.010:
                    break
        if best is None and time.perf_counter() < self._deadline():
            for dj in range(-26, 27, 3):
                for di in range(-26, 27, 3):
                    i, j = ci + di, cj + dj
                    if not g.ok(i, j):
                        continue
                    cx, cy = g.center(i, j)
                    d = math.hypot(cx - x, cy - y)
                    if d <= 56 or d > 105 or (best is not None and d >= best[0]) or math.hypot(cx - zc[0], cy - zc[1]) > zr:
                        continue
                    cz = g.ground(cx, cy)
                    if all(not g.los(tx, ty, tz, cx, cy, cz) for (tx, ty, tz) in threats):
                        best = (d, cx, cy)
                    if time.perf_counter() > self._deadline() + 0.012:
                        break
        return (best[1], best[2]) if best else None

    def _steer(self, x, y, tx, ty):
        """Unit move toward (tx, ty), bent around unwalkable cells just ahead."""
        ux, uy = unit(tx - x, ty - y)
        if self.g is None or (ux == 0 and uy == 0):
            return ux, uy
        base = math.atan2(uy, ux)
        for off in (0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.9, -1.9):
            a = base + off * self.stuck_side
            ok = True
            for dist in (3.0, 6.0):
                i, j = self.g.cell(x + math.cos(a) * dist, y + math.sin(a) * dist)
                if not self.g.ok(i, j) and self.g.ok(*self.g.cell(x, y)):
                    ok = False
                    break
            if ok:
                return math.cos(a), math.sin(a)
        return ux, uy

    def _lead(self, x, y, z, m):
        tx, ty, tz = m["pos"]
        vx, vy, vz = m.get("vel", [0, 0, 0])
        t = 0.0
        for _ in range(2):
            px, py = tx + vx * t, ty + vy * t
            d = math.hypot(px - x, py - y)
            t = d / self.gun_speed
        px, py, pz = tx + vx * t, ty + vy * t, tz + vz * t
        d = math.hypot(px - x, py - y)
        return [math.atan2(py - y, px - x), math.atan2(pz + 4.5 - (z + 7.0), max(d, 1.0))], d

    # ------------------------------------------------------------------ main
    def act(self, obs):
        self.t0 = time.perf_counter()
        self.now = obs.get("t", 0.0)
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs.get("t", 0.0)
        zone = obs["zone"]
        act = {}

        # offers: best upgrade we like
        off = obs.get("offer")
        if off:
            ids = [c.get("id") for c in off.get("choices", [])]
            pick = 0
            for pref in UPGRADE_PREF:
                if pref in ids:
                    pick = ids.index(pref)
                    break
            act["pick"] = pick

        # --- events
        for ev in obs.get("events") or []:
            typ = ev.get("type")
            if typ == "damaged":
                wpn = str(ev.get("weapon"))
                self.stats["dmg"][wpn] = round(self.stats["dmg"].get(wpn, 0) + float(ev.get("amount") or 0))
                if ev.get("by") is not None:
                    b = float(ev.get("bearing") or 0.0)
                    self.attackers[ev["by"]] = (t, x + math.cos(b) * 120, y + math.sin(b) * 120)
                    self.last_hit_t = t
            elif typ == "hit":
                self.last_dealt_t = t
        for sd in obs.get("sounds") or []:
            if sd.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet"):
                self.sounds.append((t - float(sd.get("age", 0)), float(sd["bearing"]), sd["kind"], sd.get("dist")))
        self.sounds = [s for s in self.sounds if t - s[0] < 4.0][-20:]

        mechs = [m for m in obs.get("mechs") or [] if m.get("kind") == "mech" and m.get("id") != me.get("id")]
        npcs = [m for m in obs.get("mechs") or [] if m.get("kind") == "npc"]
        for m in mechs:
            if m["id"] in self.attackers:
                self.attackers[m["id"]] = (self.attackers[m["id"]][0], m["pos"][0], m["pos"][1])
        self.attackers = {k: v for k, v in self.attackers.items() if t - v[0] < 3.0}

        # --- threat assessment
        threats = []           # (x, y, z) positions to hide from
        hot = False
        quiet_relocate = False
        for m in mechs:
            mx, my_, mz = m["pos"]
            d = math.hypot(mx - x, my_ - y)
            yaw_to_me = math.atan2(y - my_, x - mx)
            facing = abs(wrap(m.get("aim", [0, 0])[0] - yaw_to_me)) < self.fov / 2 + 0.15
            attacking = m["id"] in self.attackers
            if attacking or (m.get("charging") and facing) or d < self.near_range + 10:
                hot = True
                threats.append((mx, my_, mz))
            elif d < self.vis_range and facing:
                quiet_relocate = True
                threats.append((mx, my_, mz))
        for k, (ta, ax_, ay_) in self.attackers.items():
            if not any(m["id"] == k for m in mechs):
                threats.append((ax_, ay_, self.g.ground(ax_, ay_) if self.g else z))
                hot = True
        for n in npcs:
            nx_, ny_, nz_ = n["pos"]
            if math.hypot(nx_ - x, ny_ - y) < 70:
                threats.append((nx_, ny_, nz_))
                quiet_relocate = True

        endgame = zone["next_radius"] <= 30 or obs.get("alive", 16) <= 2
        in_zone = me.get("in_zone", True)

        # --- zone / hideout bookkeeping
        zkey = (round(zone["next_center"][0], 1), round(zone["next_center"][1], 1), round(zone["next_radius"], 1))
        drops = [tuple(d["pos"][:2]) for d in obs.get("drops") or []]
        if self.g is not None and not endgame:
            need = (zkey != self.zone_key or self.hide is None
                    or not self._in_next(self.hide[0], self.hide[1], zone, 5.0))
            if self.hide is None and zkey == self.zone_key and self.no_path_n >= 3 and t - self.no_path_t < 5.0:
                need = False
            if need and time.perf_counter() < self._deadline():
                c = self._choose_hide(x, y, zone, [tt for tt in threats], drops)
                self.zone_key = zkey
                if c is not None:
                    hx, hy = self.g.center(*c)
                    self.hide = (hx, hy)
                    self.hide_aim_due = True
                    self._plan_to(x, y, hx, hy)

        if getattr(self, "hide_aim_due", False) and self.hide and time.perf_counter() < self.t0 + 0.010:
            self.hide_aim = self._open_dir(self.hide[0], self.hide[1], 0.0)
            self.hide_aim_due = False

        # --- continue any path planning in the remaining slice
        if self.planner is not None and not self.planner.done:
            self.planner.step(self._deadline())
            if self.planner.done:
                p = self.planner.path
                if p:
                    self.path = [self.g.center(i, j) for (i, j) in p]
                    self.path_i = 0
                else:
                    gi = self.pending_goal
                    if gi:
                        self.bad_hides.add(gi)
                        self.hide = None
                        self.no_path_n += 1
                        self.no_path_t = obs.get("t", 0.0)
                self.planner = None

        # --- choose mode
        if endgame:
            mode = "FIGHT"
        elif hot or (self.cover is not None and t - self.cover_t < 2.0):
            mode = "EVADE"
        elif not in_zone:
            mode = "ZONE"
        elif quiet_relocate:
            mode = "AVOID"
        elif self.hide is not None and math.hypot(self.hide[0] - x, self.hide[1] - y) > 3.0:
            mode = "TRAVEL"
        elif self.hide is None and not self._in_next(x, y, zone, 8.0):
            mode = "TRAVEL"
        else:
            mode = "HIDE"
        self.mode = mode
        self.stats["modes"][mode] = self.stats["modes"].get(mode, 0) + 1

        move = [0.0, 0.0]
        boost = False
        jump = False
        aim = None
        fire0 = False
        fire1 = False

        # target for aiming: nearest visible attacker > nearest visible mech
        target = None
        if mechs:
            atk = [m for m in mechs if m["id"] in self.attackers]
            pool = atk or mechs
            target = min(pool, key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))

        if mode in ("EVADE", "AVOID") and self.g is not None and threats:
            if self.cover is None or t - self.cover_t > 0.6 or math.hypot(self.cover[0] - x, self.cover[1] - y) < 2.5:
                c = self._cover_from(x, y, z, threats, zone)
                if c is not None:
                    self.cover, self.cover_t = c, t
            if self.cover is not None and math.hypot(self.cover[0] - x, self.cover[1] - y) > 2.5:
                mvx, mvy = self._steer(x, y, self.cover[0], self.cover[1])
                move = [mvx, mvy]
                boost = mode == "EVADE" and me.get("energy", 0) > 15
            elif mode == "EVADE":
                # no cover in reach: run for the (hidden) hideout along the path, else away from the threats
                if self.hide is not None and math.hypot(self.hide[0] - x, self.hide[1] - y) > 6:
                    wx, wy = self._waypoint(x, y, self.hide[0], self.hide[1])
                else:
                    cx_ = sum(q[0] for q in threats) / len(threats)
                    cy_ = sum(q[1] for q in threats) / len(threats)
                    ux, uy = unit(x - cx_, y - cy_)
                    wx, wy = x + ux * 30, y + uy * 30
                move = list(self._steer(x, y, wx, wy))
                boost = me.get("energy", 0) > 5
            if mode == "AVOID" and (self.cover is None or math.hypot(self.cover[0] - x, self.cover[1] - y) <= 2.5):
                mode = "TRAVEL" if self.hide is not None and math.hypot(self.hide[0] - x, self.hide[1] - y) > 3.0 else "HIDE"
        elif mode == "EVADE":
            if target is not None:
                tx, ty = target["pos"][0], target["pos"][1]
                ux, uy = unit(x - tx, y - ty)
                move = [ux, uy]
                boost = True

        if mode in ("TRAVEL", "ZONE"):
            gx_, gy_ = (self.hide if self.hide else tuple(zone["next_center"]))
            wx, wy = self._waypoint(x, y, gx_, gy_)
            mvx, mvy = self._steer(x, y, wx, wy)
            move = [mvx, mvy]
            if self.no_path_t > t - 20.0 and not self.path:
                # boxed in (no walkable route): hop and jet toward the goal
                move = list(unit(gx_ - x, gy_ - y))
                jump = bool(me.get("grounded"))
                boost = not me.get("grounded") or me.get("energy", 0) > 60
            if mode == "ZONE":
                boost = me.get("energy", 0) > 10
            else:
                # boost only if we would otherwise arrive after the shrink ends
                dist = math.hypot(gx_ - x, gy_ - y)
                t_left = zone.get("shrink_end", 1e9) - t
                boost = (not self._in_next(x, y, zone, 2.0)) and dist / max(self.walk_speed, 1.0) > 0.8 * t_left
        if mode == "FIGHT":
            ncx, ncy = zone["next_center"]
            dcen = math.hypot(ncx - x, ncy - y)
            limit = max(3.0, zone["next_radius"] * 0.6)
            if target is not None:
                tx, ty = target["pos"][0], target["pos"][1]
                ux, uy = unit(tx - x, ty - y)
                side = 1.0 if (int(t / 1.0) % 2) else -1.0
                vx, vy = -uy * side, ux * side
                if dcen > limit:
                    cx, cy = unit(ncx - x, ncy - y)
                    vx, vy = vx + 1.5 * cx, vy + 1.5 * cy
                move = list(unit(vx, vy))
                boost = me.get("energy", 0) > 40
            elif dcen > limit or not in_zone:
                wx, wy = self._waypoint(x, y, ncx, ncy) if self.g else (ncx, ncy)
                move = list(self._steer(x, y, wx, wy))
                boost = not in_zone

        # --- stuck handling (any mode that asked to move)
        if abs(move[0]) + abs(move[1]) > 0.3:
            if self.stuck_ref is None or t - self.stuck_ref[2] > 1.5:
                if self.stuck_ref is not None and math.hypot(x - self.stuck_ref[0], y - self.stuck_ref[1]) < 2.0:
                    self.stuck_until = t + 1.0
                    self.stuck_side = -self.stuck_side
                    if self.hide and mode == "TRAVEL":
                        self._plan_to(x, y, self.hide[0], self.hide[1])
                self.stuck_ref = (x, y, t)
            if t < self.stuck_until:
                a = math.atan2(move[1], move[0]) + 1.2 * self.stuck_side
                move = [math.cos(a), math.sin(a)]
                jump = bool(me.get("grounded"))
                boost = True
        else:
            self.stuck_ref = None

        # --- aiming & shooting
        weapons = me.get("weapons") or [None, None]
        gun = weapons[0] if weapons and weapons[0] and weapons[0].get("kind") == "gun" else None
        laser_slot = None
        for k, wp in enumerate(weapons):
            if wp and wp.get("kind") == "laser":
                laser_slot = k
        if target is not None:
            aim, d = self._lead(x, y, z, target)
            err = abs(wrap(me.get("aim", [0, 0])[0] - aim[0]))
            being_hit = t - self.last_hit_t < 1.5 and target["id"] in self.attackers
            can_shoot = mode == "FIGHT" or being_hit or (mode == "EVADE" and d < self.near_range)
            if can_shoot and d < self.gun_range and err < 0.03 + 2.5 / max(d, 1.0):
                fire0 = True
            if laser_slot is not None and mode == "FIGHT" and d < self.laser_range:
                lw = weapons[laser_slot]
                ch = float(lw.get("charge") or 0.0)
                cd = float(lw.get("cooldown") or 0.0)
                if cd <= 0.0:
                    if ch < self.laser_charge - 0.02 or err > 1.5 / max(d, 1.0):
                        fire1 = True          # keep charging (or hold until aligned)
                    else:
                        fire1 = False         # release = fire
                if laser_slot == 0:
                    fire0, fire1 = fire1, False
        if aim is None:
            if mode == "HIDE":
                # scan: look at recent gunfire, else sweep the open side in 120-degree steps
                if self.sounds:
                    aim = [self.sounds[-1][1], 0.0]
                else:
                    if t - self.scan_t > 2.5:
                        self.scan_t, self.scan_k = t, (self.scan_k + 1) % 3
                    aim = [self.hide_aim + (self.scan_k - 1) * (self.fov * 0.9), 0.0]
            elif self.attackers:
                k = max(self.attackers, key=lambda q: self.attackers[q][0])
                _, ax_, ay_ = self.attackers[k]
                aim = [math.atan2(ay_ - y, ax_ - x), 0.0]
            elif abs(move[0]) + abs(move[1]) > 0.1:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = list(me.get("aim", [0.0, 0.0]))
        if fire0 or fire1:
            self.stats["shots"] += 1
        # reload when empty and calm
        if gun and gun.get("ammo", 1) == 0 and not gun.get("reload"):
            act["reload"] = True

        # --- pickups: take a laser if one is in reach and slot 1 is free
        near = [p for p in obs.get("pickups") or []
                if p.get("kind") == "laser" and math.dist(p["pos"][:2], (x, y)) < self.reach - 0.3
                and abs(p["pos"][2] - z) < 3.8]
        if near and weapons[1] is None and laser_slot is None:
            act["take"] = near[0]["id"]

        n = math.hypot(move[0], move[1])
        if n > 1.0:
            move = [move[0] / n, move[1] / n]
        act.update(move=[float(move[0]), float(move[1])], aim=[float(aim[0]), float(aim[1])],
                   boost=bool(boost), jump=bool(jump))
        if fire0 or fire1:
            act["fire"] = [bool(fire0), bool(fire1)]

        if self.hide:
            mf.draw.point([self.hide[0], self.hide[1], z], "#40c0ff", 6)
        mf.draw.label(f"{mode} hp{me['hp']:.0f} thr{len(threats)}")
        if obs.get("tick", 0) % 600 == 0:
            mf.log(f"ghost t={t:.0f} alive={obs.get('alive')} hp={me['hp']:.0f} {self.stats}")
        return act

    def _plan_to(self, x, y, gx_, gy_):
        if self.g is None:
            return
        s = self.g.snap(*self.g.cell(x, y))
        gcell = self.g.snap(*self.g.cell(gx_, gy_))
        self.path = None
        if s is None or gcell is None:
            self.planner = None
            return
        self.pending_goal = gcell
        self.planner = AStar(self.g, s, gcell)
        self.planner.step(self._deadline())
        if self.planner.done:
            p = self.planner.path
            self.path = [self.g.center(i, j) for (i, j) in p] if p else None
            self.path_i = 0
            if not p:
                self.bad_hides.add(gcell)
                self.hide = None
                self.no_path_n += 1
                self.no_path_t = self.now
            self.planner = None

    def _waypoint(self, x, y, gx_, gy_):
        """Furthest path point within 24 m that is straight-line walkable; else the goal."""
        if not self.path:
            return gx_, gy_
        p = self.path
        # advance past points we've reached
        while self.path_i < len(p) - 1 and math.hypot(p[self.path_i][0] - x, p[self.path_i][1] - y) < 4.0:
            self.path_i += 1
        # re-sync if we drifted: nearest point ahead
        best_k = self.path_i
        best_d = math.hypot(p[best_k][0] - x, p[best_k][1] - y)
        for k in range(self.path_i, min(len(p), self.path_i + 12)):
            d = math.hypot(p[k][0] - x, p[k][1] - y)
            if d < best_d:
                best_k, best_d = k, d
        self.path_i = best_k
        wp = p[self.path_i]
        for k in range(self.path_i, min(len(p), self.path_i + 8)):
            if math.hypot(p[k][0] - x, p[k][1] - y) > 24:
                break
            if self.g.walk_line(x, y, p[k][0], p[k][1]):
                wp = p[k]
        return wp


if __name__ == "__main__":
    mf.run(Bot)
