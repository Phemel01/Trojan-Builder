"""OpeningLoop (4dcef84) + lead aiming.

Stock OpeningLoop (openingloop.py, unchanged) decides what to do. This layer changes only the final aim:
  * gun:    aim where the target will be when the bullet arrives (travel time = distance / gun speed
            from the weapon's own stats, plus half a decision of delay), using the target's velocity;
  * laser:  aim ~one decision ahead (the beam is instant but our torso turns after we decide);
  * mortar: solve the ballistic pitch (shell speed, gravity, tube height) for the led target distance;
            on those decisions the gun is held, because one torso cannot aim at two pitches.
Targets are the mech/NPC/crate that V4 is already aiming at (closest to V4's commanded yaw).
"""
import math

import mechfight as mf
from openingloop import Bot as V4  # OpeningLoop 4dcef84, unchanged


def _yaw(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


class Bot(V4):
    name = "Trojan OpeningLoop lead"

    def _target(self, obs, yaw, pos):
        best = None
        pool = [("mech", m) for m in obs.get("mechs", ())] + [("crate", c) for c in obs.get("crates", ())]
        for kind, it in pool:
            p = it["pos"]
            d = math.hypot(p[0] - pos[0], p[1] - pos[1])
            if d < 1.0 or d > 300:
                continue
            err = abs(math.remainder(_yaw(pos, p) - yaw, 2 * math.pi))
            if err < 0.06 and (best is None or err < best[0]):
                best = (err, kind, it)
        return best

    def _mortar_pitch(self, d, dz, ws, muzzle):
        v, g = float(ws.get("speed", 72)), float(ws.get("gravity", 22))
        h = muzzle + 2.5 - dz                     # tube above target centre
        best = None
        p = -0.6
        while p <= 1.25:
            vx, vz = v * math.cos(p), v * math.sin(p)
            disc = vz * vz + 2 * g * h
            if disc >= 0 and vx > 1e-6:
                t = (vz + math.sqrt(disc)) / g
                err = abs(vx * t - d)
                if best is None or err < best[0]:
                    best = (err, p, t)
            p += 0.01
        return best

    def act(self, obs):
        act = super().act(obs)
        try:
            self._lead(obs, act)
        except Exception as e:      # never let the aim layer break a decision
            mf.log(f"lead layer error: {e!r}")
        return act

    def _lead(self, obs, act):
        fire = act.get("fire")
        if not fire or not any(fire) or "aim" not in act:
            return
        me = obs["self"]
        pos = me["pos"]
        found = self._target(obs, act["aim"][0], pos)
        if not found:
            return
        _, kind, tgt = found
        tv = tgt.get("vel") or [0.0, 0.0, 0.0]
        tp = tgt["pos"]
        weapons = me.get("weapons") or []
        stats = me.get("stats", {})
        muzzle = stats.get("muzzle_height", 7.0)
        height = 4.5 if tgt.get("kind") == "mech" else (1.75 if tgt.get("kind") == "npc" else 1.2)
        slots = {w["kind"]: i for i, w in enumerate(weapons) if w}
        mortar_i = slots.get("mortar")
        d0 = math.hypot(tp[0] - pos[0], tp[1] - pos[1])
        if mortar_i is not None and mortar_i < len(fire) and fire[mortar_i] and weapons[mortar_i].get("cooldown", 1) <= 0:
            ws = weapons[mortar_i].get("stats") or self.cfg.get("weapons", {}).get("mortar", {})
            t = 0.0
            sol = None
            for _ in range(3):
                px, py = tp[0] + tv[0] * (t + 0.1), tp[1] + tv[1] * (t + 0.1)
                d = math.hypot(px - pos[0], py - pos[1])
                sol = self._mortar_pitch(d, tp[2] - pos[2], ws, muzzle + 2.5)
                if sol is None:
                    break
                t = sol[2]
            if sol is not None and sol[0] < 3.0:
                act["aim"] = [_yaw(pos, (px, py)), sol[1]]
                fire = list(fire)
                for k, w in enumerate(weapons):
                    if w and w["kind"] == "gun" and k < len(fire):
                        fire[k] = False
                act["fire"] = fire
                return
        laser_i = slots.get("laser")
        if laser_i is not None:
            lw = weapons[laser_i]
            lws = lw.get("stats") or self.cfg.get("weapons", {}).get("laser", {})
            releasing = lw.get("cooldown", 1) <= 0 and lw.get("charge", 0) >= float(lws.get("charge", 0.75)) - 0.12
            if releasing:
                t = 0.1
                px, py = tp[0] + tv[0] * t, tp[1] + tv[1] * t
                d = math.hypot(px - pos[0], py - pos[1])
                act["aim"] = [_yaw(pos, (px, py)), math.atan2(tp[2] + height - pos[2] - muzzle, max(0.1, d))]
                return
        gun_i = slots.get("gun")
        if gun_i is not None and gun_i < len(fire) and fire[gun_i]:
            ws = weapons[gun_i].get("stats") or self.cfg.get("weapons", {}).get("gun", {})
            v = float(ws.get("speed", 252.0))
            t = d0 / v + 0.05
            for _ in range(3):
                px, py = tp[0] + tv[0] * t, tp[1] + tv[1] * t
                d = math.hypot(px - pos[0], py - pos[1])
                t = d / v + 0.05
            pz = tp[2] + tv[2] * t + height
            act["aim"] = [_yaw(pos, (px, py)), math.atan2(pz - pos[2] - muzzle, max(0.1, d))]
            return
        if laser_i is not None:
            t = 0.1
            px, py = tp[0] + tv[0] * t, tp[1] + tv[1] * t
            d = math.hypot(px - pos[0], py - pos[1])
            act["aim"] = [_yaw(pos, (px, py)), math.atan2(tp[2] + height - pos[2] - muzzle, max(0.1, d))]


if __name__ == "__main__":
    mf.run(Bot)
