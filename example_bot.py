"""Mech Royale variant with zone positioning and obstacle recovery."""
import math
import mechfight as mf


class Bot:
    name = "trojan-zone-v1"

    def __init__(self, hello: mf.Hello):
        self.reach = hello["config"]["loot"]["pickup_radius"]

        self.progress_origin = None
        self.progress_tick = 0
        self.detour_until = 0
        self.detour_side = -1

    def movement(self, obs, x, y):
        zone = obs["zone"]
        me = obs["self"]
        tick = obs["tick"]
        cx, cy = zone["next_center"]
        dx, dy = cx - x, cy - y
        distance = math.hypot(dx, dy)
        # Walk only far enough into the next circle to leave a buffer.
        holding_radius = max(0.0, zone["next_radius"] - 35.0)
        remaining = distance - holding_radius
        yaw = math.atan2(dy, dx)
        if remaining <= 1.0 and me["in_zone"]:
            self.progress_origin = None
            self.detour_until = 0
            return [0.0, 0.0], yaw, False, False

        jump = False
        if self.progress_origin is None:
            self.progress_origin = (x, y)
            self.progress_tick = tick
        elif tick - self.progress_tick >= 10:
            if math.dist(self.progress_origin, (x, y)) < 2.0:
                self.detour_until = tick + 14
                self.detour_side *= -1
                jump = me["grounded"]
            self.progress_origin = (x, y)
            self.progress_tick = tick
        detouring = tick < self.detour_until
        if detouring:
            yaw += self.detour_side * math.pi / 3.0
        speed = 1.0 if detouring or not me["in_zone"] else min(1.0, remaining / 8.0)
        return (
            [speed * math.cos(yaw), speed * math.sin(yaw)],
            yaw,
            not me["in_zone"] and me["grounded"] and not detouring,
            jump,
        )

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        goal = obs["zone"]["next_center"]
        act: mf.Action = {"tick": obs["tick"]}
        if obs["offer"]:
            act["pick"] = 0
        near = [p for p in obs["pickups"] if math.dist(p["pos"][:2], (x, y)) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            act["take"] = near[0]["id"]
        enemies = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"], me["pos"]))
        move, yaw, boost, jump = self.movement(obs, x, y)
        act.update(move=move, aim=[yaw, 0.0], boost=boost, jump=jump)
        if enemies:
            t = enemies[0]["pos"]
            aim = [math.atan2(t[1] - y, t[0] - x), math.atan2(t[2] - z - 2.5, math.dist(t[:2], (x, y)))]
            act.update(aim=aim, fire=[True, True])
            mf.draw.line([x, y, z + 7], [t[0], t[1], t[2] + 4.5], "#ff5040")
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        mf.draw.label(f"hp {me['hp']:.0f} · {len(enemies)} in sight")
        return act


mf.run(Bot)
