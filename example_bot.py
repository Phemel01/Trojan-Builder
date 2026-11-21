"""Trojan Evade: official baseline routing + a threat/health state machine.

Movement toward the zone is the official starter's, unchanged (head for next_center at full speed,
boost outside the zone, jump every 40 ticks). The only behaviour change is how the bot reacts to
opponents:

  ROAM     no threat: baseline routing. Visible mechs that are far away and not engaging are left
           alone (the starter shot at anything in sight, which starts fights it did not need).
  EVADE    a threat is present: move away from it (weighted by 1/distance), zig-zag sideways, boost
           while energy allows, keep the torso on the attacker and fire back. Move and aim are
           independent in mechfight/1, so retreating does not mean going quiet.
  RECOVER  hp is low and the threat has gone: keep moving inside the zone, do not open fire, and
           wait for hp to come back if the rules regenerate it (read from hello.config), else for a
           fixed cool-down.

A threat is any of: a `damaged` event from a mech in the last 2 s, a visible mech inside the
danger radius, two or more visible mechs inside 1.5x that radius, any visible mech while hp is
low, a visible laser charging toward us, or a projectile whose closest approach is near us soon.

Zone safety beats evasion: outside the current zone the bot heads in and only sidesteps; near the
next circle's edge it drops the outward part of the escape vector and slides along the edge.

Cost per tick: one pass over visible mechs, projectiles, beams and events (O(n), n is tens at
most); no map access, no allocation beyond small lists. Well under 1 ms in CPython.
"""
import math
import mechfight as mf

TAU = 2.0 * math.pi


def _wrap(a):
    return (a + math.pi) % TAU - math.pi


def _unit(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)


class Bot:
    name = "trojan-evade"

    # Tunables (seconds are converted to ticks at 10 Hz).
    THREAT_MEMORY_S = 2.0      # how long a hit keeps us in EVADE
    EVADE_HOLD_S = 1.0         # hysteresis after the last threat disappears
    LOW_HP = 0.45              # fraction of max_hp that counts as "low"
    RECOVER_TO = 0.85          # leave RECOVER at this fraction when regen exists
    RECOVER_MAX_S = 6.0        # ...or after this long either way
    ENERGY_RESERVE = 0.30      # keep this fraction of energy unboosted while evading
    ZIG_PERIOD_S = 0.9         # sideways zig-zag half-period
    DODGE_RADIUS = 3.5         # m: projectile closest approach that triggers a sidestep
    DODGE_HORIZON_S = 1.2

    def __init__(self, hello: mf.Hello):
        cfg = hello.get("config", {}) or {}
        self.reach = cfg.get("loot", {}).get("pickup_radius", 3.0)
        hz = (cfg.get("match", {}) or {}).get("tick_hz") or (cfg.get("bot", {}) or {}).get("hz") or 10.0
        self.dt = 1.0 / float(hz) if hz else 0.1
        vision = cfg.get("vision", {}) or {}
        near = float(vision.get("near_range") or 20.0)
        rng = float(vision.get("range") or 80.0)
        # Danger radius: assumption, not a rule. Scale off the vision numbers so it follows the config.
        self.danger_r = max(25.0, min(60.0, 1.5 * near, 0.6 * rng))
        # Return-fire range: anyone engaging us inside this gets shot back at (don't take free hits).
        gun = (cfg.get("weapons", {}) or {}).get("gun", {}) or {}
        self.shoot_r = float(gun.get("range") or gun.get("max_range") or 0.85 * rng)
        self.attackers = {}              # mech id -> last tick it damaged us
        self.dodge_side = 1.0
        self.dodge_until = -1
        mech = cfg.get("mech", {}) or {}
        self.speed = float(mech.get("walk_speed", 11.7))
        self.regen = float(mech.get("regen_per_s", 0.0))
        self.state = "ROAM"
        self.last_threat_tick = -10**9
        self.threat_bearings = []        # (tick, yaw toward attacker)
        self.recover_since = None
        self.recover_spent = False      # timed out without regen: don't re-enter until hp recovers
        self.zig = 1.0
        self.zig_tick = 0
        self.prev_hp = None
        self.state_ticks = {"ROAM": 0, "EVADE": 0, "RECOVER": 0}
        self.dmg_by = {}                 # weapon -> damage taken (for the stderr summary)
        mf.log(f"trojan-evade: danger_r={self.danger_r:.0f}m regen={self.regen} "
               f"mech_keys={sorted(mech.keys())}")

    # ---------- threat assessment ----------
    def _threats(self, obs, x, y, z, hp_frac):
        tick = obs["tick"]
        me = obs["self"]
        my_id = me.get("id")
        vecs = []          # (ux, uy, weight) pointing AWAY from danger
        dodge = None       # sidestep unit vector for an incoming shot
        engaged = []       # mechs worth shooting back at
        hot_ids = set()

        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("by") is not None and ev.get("bearing") is not None:
                self.threat_bearings.append((tick, float(ev["bearing"])))
                self.attackers[ev["by"]] = tick
        horizon = int(self.THREAT_MEMORY_S / self.dt)
        self.attackers = {k: v for k, v in self.attackers.items() if tick - v <= horizon}
        self.threat_bearings = [(t, b) for t, b in self.threat_bearings if tick - t <= horizon]
        for _, b in self.threat_bearings:
            vecs.append((-math.cos(b), -math.sin(b), 1.0))

        mechs = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech" and m.get("id") != my_id]
        close = 0
        for m in mechs:
            mx, my_, mz = m["pos"]
            d = math.hypot(mx - x, my_ - y)
            ux, uy = _unit(x - mx, y - my_)
            yaw_to_me = math.atan2(y - my_, x - mx)
            aim = m.get("aim") or [0.0, 0.0]
            facing = abs(_wrap(aim[0] - yaw_to_me)) < math.radians(20)
            if d < 1.5 * self.danger_r:
                close += 1
            hot = d < self.danger_r or (facing and d < 1.5 * self.danger_r) or (hp_frac < self.LOW_HP)
            if m.get("charging") and facing:
                hot = True
                if tick > self.dodge_until:          # pick one side per charge and commit to it
                    self.dodge_side = self.zig
                self.dodge_until = tick + 3
                px, py = -math.sin(yaw_to_me), math.cos(yaw_to_me)
                dodge = (px * self.dodge_side, py * self.dodge_side)
            if hot:
                vecs.append((ux, uy, max(0.3, self.danger_r / max(d, 1.0))))
                hot_ids.add(m.get("id"))
            # Shoot back at anyone engaging us in range, whether or not we are fleeing them. Proximity
            # alone is not a reason to open fire (that starts fights with mechs that were passing by).
            if d < self.shoot_r and (facing or m.get("id") in self.attackers or d < 0.5 * self.danger_r):
                engaged.append((d, m))
        if close >= 2:
            for m in mechs:
                mx, my_, _ = m["pos"]
                d = math.hypot(mx - x, my_ - y)
                if d < 1.5 * self.danger_r and m.get("id") not in hot_ids:
                    ux, uy = _unit(x - mx, y - my_)
                    vecs.append((ux, uy, 0.5))
                    if all(e[1] is not m for e in engaged):
                        engaged.append((d, m))

        for p in obs.get("projectiles") or []:
            if p.get("kind") == "npc":
                continue
            px, py, _ = p["pos"]
            vx, vy, _ = p["vel"]
            v2 = vx * vx + vy * vy
            if v2 < 1e-6:
                continue
            rx, ry = x - px, y - py
            t = (rx * vx + ry * vy) / v2
            if 0.0 < t < self.DODGE_HORIZON_S:
                cx, cy = px + vx * t - x, py + vy * t - y
                if math.hypot(cx, cy) < self.DODGE_RADIUS:
                    sx, sy = _unit(-vy, vx)
                    if sx * -cx + sy * -cy < 0:   # step to the side away from the predicted miss point
                        sx, sy = -sx, -sy
                    dodge = (sx, sy)

        for b in obs.get("beams") or []:
            if b.get("by") is not None and b.get("age", 1.0) < 0.3:
                path = b.get("path") or []
                if path:
                    bx, by_, _ = path[0]
                    if math.hypot(bx - x, by_ - y) < 1.5 * self.danger_r:
                        ux, uy = _unit(x - bx, y - by_)
                        vecs.append((ux, uy, 0.7))

        engaged.sort(key=lambda e: e[0])
        return vecs, dodge, engaged

    # ---------- zone-constrained movement ----------
    def _constrain(self, obs, x, y, vx, vy):
        zone = obs["zone"]
        me = obs["self"]
        cx, cy = zone["center"]
        nx, ny = zone["next_center"]
        if not me.get("in_zone", True):
            # Outside: zone wins, keep only a small sideways part of the escape.
            ix, iy = _unit(cx - x, cy - y)
            side = vx * -iy + vy * ix
            return _unit(ix + 0.35 * side * -iy, iy + 0.35 * side * ix)
        # Inside: don't let the escape carry us past the safe ring. The ring around the next circle
        # widens with the time left before the shrink ends: we may stray only as far as we can walk
        # back (at half speed, to leave margin) before it closes.
        dn = math.hypot(nx - x, ny - y)
        t_left = max(0.0, float(zone.get("shrink_end", 0.0)) - float(obs.get("t", 0.0)))
        speed = float((me.get("stats") or {}).get("walk_speed", self.speed))
        r_safe = 0.85 * zone["next_radius"] + 0.5 * speed * t_left
        dc = math.hypot(cx - x, cy - y)
        r_cur = 0.9 * zone["radius"]
        r_safe = min(r_safe, r_cur + math.hypot(nx - cx, ny - cy)) if r_cur > 0 else r_safe
        for (ox, oy, d, r) in ((nx, ny, dn, r_safe), (cx, cy, dc, r_cur)):
            if r > 0 and d > 0.6 * r:
                rx, ry = _unit(x - ox, y - oy)           # outward radial
                out = vx * rx + vy * ry
                if out > 0:
                    k = min(1.0, (d - 0.6 * r) / (0.4 * r))
                    vx -= out * rx * k
                    vy -= out * ry * k
                    if d > r:                             # past the ring: also pull inward
                        vx -= 0.6 * rx
                        vy -= 0.6 * ry
        return _unit(vx, vy)

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        tick = obs["tick"]
        goal = obs["zone"]["next_center"]
        hp_frac = me["hp"] / me["max_hp"] if me.get("max_hp") else 1.0
        act: mf.Action = {"tick": tick}

        # --- unchanged from the official starter: offers and pickups ---
        if obs.get("offer"):
            act["pick"] = 0
        near = [p for p in obs.get("pickups") or [] if math.dist(p["pos"][:2], (x, y)) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            act["take"] = near[0]["id"]

        # --- unchanged from the official starter: routing ---
        yaw = math.atan2(goal[1] - y, goal[0] - x)
        move = [math.cos(yaw), math.sin(yaw)]
        boost = not me["in_zone"]
        jump = tick % 40 == 0
        aim = [yaw, 0.0]
        fire = None

        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged":
                w = str(ev.get("weapon"))
                self.dmg_by[w] = round(self.dmg_by.get(w, 0.0) + float(ev.get("amount") or 0.0), 1)
        vecs, dodge, engaged = self._threats(obs, x, y, z, hp_frac)
        if vecs or dodge or self.attackers:
            self.last_threat_tick = tick

        if tick - self.zig_tick >= int(self.ZIG_PERIOD_S / self.dt):
            self.zig = -self.zig
            self.zig_tick = tick

        hold = int(self.EVADE_HOLD_S / self.dt)
        prev = self.state
        if vecs or dodge or tick - self.last_threat_tick <= hold:
            self.state = "EVADE"
        elif not self.recover_spent and (hp_frac < self.LOW_HP or (
                prev in ("EVADE", "RECOVER") and hp_frac < self.RECOVER_TO and self.regen > 0)):
            if prev != "RECOVER":
                self.recover_since = tick
            self.state = "RECOVER"
            if tick - self.recover_since > int(self.RECOVER_MAX_S / self.dt):
                self.state = "ROAM"
                self.recover_spent = True
        else:
            self.state = "ROAM"
        if hp_frac >= self.RECOVER_TO or self.state == "EVADE":
            self.recover_spent = False

        if self.state == "EVADE":
            ex = sum(u * w for u, _, w in vecs)
            ey = sum(v * w for _, v, w in vecs)
            ex, ey = _unit(ex, ey)
            if ex == 0.0 and ey == 0.0:
                ex, ey = _unit(goal[0] - x, goal[1] - y)   # threat remembered but no vector: keep routing
            # zig-zag perpendicular to the escape line
            px, py = -ey * self.zig, ex * self.zig
            vx, vy = ex + 0.45 * px, ey + 0.45 * py
            if dodge:
                vx, vy = vx + 1.5 * dodge[0], vy + 1.5 * dodge[1]
            vx, vy = self._constrain(obs, x, y, vx, vy)
            if vx == 0.0 and vy == 0.0:
                vx, vy = move
            move = [vx, vy]
            boost = me["energy"] > self.ENERGY_RESERVE * me["max_energy"] or not me["in_zone"]
            jump = bool(dodge) and me.get("grounded", False)
        elif self.state == "RECOVER":
            # Keep moving (a stationary mech is the easiest target) but stay calm near the next centre.
            dx, dy = goal[0] - x, goal[1] - y
            d = math.hypot(dx, dy)
            ix, iy = _unit(dx, dy)
            tx, ty = -iy * self.zig, ix * self.zig
            inward = min(1.0, d / max(10.0, 0.5 * obs["zone"]["next_radius"]))
            vx, vy = self._constrain(obs, x, y, inward * ix + tx, inward * iy + ty)
            move = [vx, vy] if (vx or vy) else move
            boost = not me["in_zone"]
            jump = False

        # --- shooting: only at mechs that are engaging us, or anything very close ---
        if engaged:
            t = engaged[0][1]["pos"]
            aim = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] - z - 2.5, math.dist(t[:2], (x, y)))]
            fire = [True, True]
        elif self.state == "ROAM":
            vis = sorted((m for m in obs.get("mechs") or [] if m.get("kind") == "mech"),
                         key=lambda m: math.dist(m["pos"][:2], (x, y)))
            if vis and math.dist(vis[0]["pos"][:2], (x, y)) < 0.5 * self.danger_r:
                t = vis[0]["pos"]
                aim = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] - z - 2.5, math.dist(t[:2], (x, y)))]
                fire = [True, True]

        n = math.hypot(move[0], move[1])
        if n > 1.0:
            move = [move[0] / n, move[1] / n]
        act.update(move=[float(move[0]), float(move[1])], aim=[float(aim[0]), float(aim[1])],
                   boost=bool(boost), jump=bool(jump))
        if fire:
            act["fire"] = fire
        self.prev_hp = me["hp"]
        self.state_ticks[self.state] += 1
        if tick % 300 == 0:   # every 30 s: where time went and what hurt us (team-only stderr)
            mf.log(f"t={obs.get('t', 0):.0f}s alive={obs.get('alive')} hp={me['hp']:.0f} "
                   f"states={self.state_ticks} dmg_by={self.dmg_by}")

        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        if engaged:
            t = engaged[0][1]["pos"]
            mf.draw.line([x, y, z + 7], [t[0], t[1], t[2] + 4.5], "#ff5040")
        mf.draw.label(f"{self.state} hp {me['hp']:.0f} thr {len(vecs)}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
