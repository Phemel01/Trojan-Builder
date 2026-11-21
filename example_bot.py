"""Trojan Placement: zone routing, cube progression and control of game abilities."""
import math

import mechfight as mf
from arena_nav import FloorMap, Navigator, direction
from evade import Bot as Evade


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def yaw_to(a, b):
    return math.atan2(b[1]-a[1], b[0]-a[0])


class Bot(Evade):
    name = "Trojan Placement v4"

    def __init__(self, hello):
        super().__init__(hello)
        self.cfg = hello.get("config", {})
        self.slot = hello.get("slot", 0)
        self.floor = FloorMap(hello)
        self.nav = Navigator(self.floor, 1 if self.slot % 2 == 0 else -1)
        self.seen = {}
        self.objective = None
        self.phase = None
        self.anchor = None
        self.focus = None
        self.jet_until = -1.0
        self.next_jet = 0.0
        self.progress_sample = None

    def zone_goal(self, pos, zone):
        center, radius = zone["next_center"], zone["next_radius"]
        phase = (*center, radius)
        if phase != self.phase:
            self.phase = phase
            radial = direction(center, pos)
            if not any(radial):
                angle = (self.slot + 1)*2.4
                radial = (math.cos(angle), math.sin(angle))
            # Keep our own entry sector as the phase progresses. The safe disk
            # contains many valid positions besides its exact center.
            offset = min(0.64*radius, max(0.33*radius, distance(center, pos)))
            self.anchor = (center[0]+radial[0]*offset, center[1]+radial[1]*offset)
        return self.anchor

    def reachable_before_shrink(self, pos, item, service, speed, obs):
        z, now = obs["zone"], obs["t"]
        if distance(item, z["center"]) > z["radius"] - min(4, z["radius"]*0.1):
            return False
        visit_s = 1.4*distance(pos, item)/speed + service
        start = max(now, z["shrink_start"])
        end = z["shrink_end"]
        fraction = min(1, max(0, (now+visit_s-start)/max(0.01, end-start)))
        center = tuple(a+(b-a)*fraction for a,b in zip(z["center"], z["next_center"]))
        radius = z["radius"]+(z["next_radius"]-z["radius"])*fraction
        if distance(item, center) > radius-min(4, radius*0.1):
            return False
        gap = max(0, distance(item, z["next_center"])-0.85*z["next_radius"])
        return gap == 0 or visit_s + 1.4*gap/speed + 3 < end-now

    def remember(self, obs):
        now, pos = obs["t"], obs["self"]["pos"]
        visible = set()
        for kind in ("cubes", "crates", "pickups"):
            for item in obs.get(kind, ()):
                key = (kind, item["id"])
                visible.add(key)
                self.seen[key] = (item, now)
        for key, (item, seen_at) in list(self.seen.items()):
            if key not in visible and (now-seen_at > 8 or distance(pos, item["pos"]) < 10):
                del self.seen[key]
        return visible

    def equipment(self, weapons, kind, radius):
        quality = {"gun":0, "mortar":28, "lightning":104 if radius<=40 else 77, "laser":101}
        if kind not in quality or kind == "gun" or any(w and w["kind"]==kind for w in weapons):
            return 0, None
        if None in weapons:
            return quality[kind], None
        for idx, weapon in enumerate(weapons):
            if weapon["kind"] != "gun" and not weapon.get("charge",0):
                gain = quality[kind] - quality.get(weapon["kind"],0)
                if gain >= 20:
                    return gain, idx
        return 0, None

    def best_loot(self, obs, visible, hp, speed):
        me, now = obs["self"], obs["t"]
        pos = me["pos"]
        empty = any(w is None for w in me["weapons"])
        candidates = list(self.seen.items()) + [
            (("drops", d["id"]), (d, now)) for d in obs.get("drops", ())]
        best, best_value = None, -math.inf
        for key, (item, seen_at) in candidates:
            kind, _ = key
            point = item["pos"]
            d = distance(pos, point)
            floor_z = self.floor.ground(point)
            if floor_z is None:
                floor_z = pos[2]
            if abs(point[2]-floor_z) > 4 and (d > 5 or abs(point[2]-pos[2]) > 4):
                continue
            service = 0
            if kind == "cubes":
                value, reach = (110 if me.get("cubes", 0)<10 else 64), (86 if me.get("cubes", 0)<3 else 62)
            elif kind == "pickups":
                gain, _ = self.equipment(me["weapons"],item["kind"],obs["zone"]["next_radius"])
                if gain <= 0:
                    continue
                value = (48+0.65*gain) if empty else (26+0.83*gain)
                reach = 148 if now < 65 and empty else 82
            elif kind == "crates":
                if hp < 0.77 or me.get("cubes", 0) >= 10:
                    continue
                value, reach, service = 76, (110 if now < 65 and not me.get("cubes") else 52), 2.8
            else:
                radius = item.get("radius", 8)
                if item.get("holder") not in (None, self.slot):
                    continue
                if any(m.get("kind")=="mech" and distance(m["pos"], point)<radius+20 for m in obs.get("mechs", ())):
                    continue
                arrival = d/speed
                wait = max(0, item["lands_in"]-arrival)
                if wait > 6:
                    continue
                service = wait + max(0, item.get("capture_s", 4)-item.get("progress", 0))
                value, reach = 130 + (1-hp)*90 - wait*5, 100
            if d > reach or not self.reachable_before_shrink(pos, point, service, speed, obs):
                continue
            if self.floor.mask and d > 5:
                cell, here = self.floor.cell(point), self.floor.cell(pos)
                if self.floor.mask[cell] or (not self.floor.mask[here] and self.floor.components[here] != self.floor.components[cell]):
                    comp = self.floor.components[here] or None
                    free = self.floor.nearest(cell, comp, 2)
                    if free is None or distance(point, self.floor.xy(free)) > 3.5:
                        continue
            value -= 0.84*d + 4*(now-seen_at)
            if self.floor.mask and not self.floor.clear(pos, point):
                value -= 22
            if key == self.objective:
                value += 12
            if value > best_value:
                best_value, best = value, (key, item, key in visible)
        return best

    def immediate_reward(self, obs, travel, hp):
        """Take a nearby game reward during Evade without abandoning an escape route."""
        me = obs["self"]
        pos, speed = me["pos"], max(1,me.get("stats",{}).get("walk_speed",self.speed))
        options = []
        for kind in ("cubes","pickups","drops"):
            for item in obs.get(kind, ()):
                point = item["pos"]
                d = distance(pos,point)
                if d > 20 or abs(pos[2]-point[2])>4 or not self.floor.clear(pos,point):
                    continue
                if not self.reachable_before_shrink(pos,point,0,speed,obs):
                    continue
                if kind == "drops":
                    remaining = item.get("capture_s",4)-item.get("progress",0)
                    if (item.get("holder") != self.slot or item["lands_in"] or remaining>1.8
                            or any(m.get("kind")=="mech" and distance(m["pos"],point)<22 for m in obs.get("mechs",()))):
                        continue
                    value = 160 + 40*(1-hp)
                else:
                    toward = direction(pos,point)
                    alignment = toward[0]*travel[0] + toward[1]*travel[1]
                    if d>6 and alignment < 0.18:
                        continue
                    if kind == "pickups":
                        gain,_ = self.equipment(me["weapons"],item["kind"],obs["zone"]["next_radius"])
                        if gain < 20:
                            continue
                        value = gain
                    else:
                        value = 92 if me.get("cubes",0)<10 else 50
                    value -= 2*d
                options.append((value,item["id"],kind,item))
        if options:
            _, ident, kind, item = max(options,key=lambda row:row[:2])
            return (kind,ident),item,True
        return None

    def pick_upgrade(self, choices, hp):
        order = {"plating":96, "armor":91, "nanites":88, "servos":84, "gun_damage":74,
                 "gun_rate":72, "laser_charge":71, "laser_damage":70, "gun_mag":69,
                 "lightning_range":67, "jets":64, "reactor":60, "capacitor":58, "sensors":56}
        if self.regen and hp < 0.65:
            order["nanites"] += 11
        return max(range(len(choices)), key=lambda i: order.get(choices[i].get("id"), 40))

    def game_triggers(self, obs, act, distance_to_target, finish=False, fleeing=False):
        me = obs["self"]
        firing = []
        for weapon in me["weapons"]:
            if not weapon:
                firing.append(False)
                continue
            kind = weapon["kind"]
            stats = weapon.get("stats") or self.cfg.get("weapons", {}).get(kind, {})
            d = distance_to_target
            if kind == "laser":
                # This game slot fires on release. Holding a full charge slows the avatar.
                charge = weapon.get("charge", 0)
                allow = d <= stats.get("range", 280) and (finish or (not fleeing and me["in_zone"]))
                firing.append(allow and weapon.get("cooldown",0) <= 0 and charge+1e-6 < stats.get("charge",0.75))
            elif kind == "lightning":
                firing.append(d <= stats.get("range", 45) and me["energy"] > stats.get("energy",3)+8)
            elif kind == "mortar":
                # Use Evade's existing direction only at short game ranges. Preserve HP
                # around nearby geometry and avoid wasting the slot at point-blank range.
                firing.append(18 < d < 58 and (not self.floor.mask or self.floor.clear(
                    me["pos"], (me["pos"][0]+12*math.cos(act["aim"][0]), me["pos"][1]+12*math.sin(act["aim"][0])))))
            else:
                firing.append(d <= stats.get("range", self.shoot_r))
        act["fire"] = firing

    def act(self, obs):
        act = super().act(obs)
        recover_move = act["move"] if self.state == "RECOVER" else None
        me, z, now = obs["self"], obs["zone"], obs["t"]
        pos, tick = me["pos"], obs["tick"]
        hp = me["hp"]/max(1, me["max_hp"])
        speed = max(1, me.get("stats", {}).get("walk_speed", self.speed))
        goal = self.zone_goal(pos, z)
        visible = self.remember(obs)
        if obs.get("offer") and obs["offer"]["choices"]:
            act["pick"] = self.pick_upgrade(obs["offer"]["choices"], hp)
        act.pop("take", None)
        available = []
        for item in obs.get("pickups", ()):
            d = distance(pos,item["pos"])
            if d <= self.reach and abs(item["pos"][2]-pos[2]) <= 4:
                gain, replace = self.equipment(me["weapons"],item["kind"],z["next_radius"])
                if gain > 0:
                    available.append((gain, -d, item["id"], replace))
        if available:
            _, _, act["take"], replace = max(available)
            if replace is not None:
                act["drop"] = replace
        gap = max(0, distance(pos, z["next_center"])-0.85*z["next_radius"])
        urgent = not me["in_zone"] or (gap > 0 and 1.5*gap/speed+4 >= z["shrink_end"]-now)
        evading = self.state == "EVADE"
        loot = None
        if not evading and not urgent:
            loot = self.best_loot(obs, visible, hp, speed)
        if evading and not urgent:
            loot = self.immediate_reward(obs,act["move"],hp)
        self.objective = loot[0] if loot else None
        if loot:
            goal = loot[1]["pos"][:2]
        if urgent or not evading or loot:
            act["move"] = list(self.nav.toward(pos[:2], tuple(goal), tick))
            act["jump"] = False
            act["boost"] = bool(me.get("grounded", True) and (urgent or (
                loot and distance(goal,pos)>22 and me["energy"]>0.78*me["max_energy"])))
            hold = min(3, loot[1].get("radius",8)*0.40) if loot and loot[0][0]=="drops" else 1.1
            if distance(goal,pos)<hold:
                act["move"], act["boost"] = [0.0,0.0], False
            if not act.get("fire"):
                if distance(goal,pos)>6:
                    act["aim"] = [yaw_to(pos,goal)+1.0*math.sin(0.72*now), 0.0]
                else:
                    act["aim"] = [(now*1.15+self.slot)%(2*math.pi)-math.pi, 0.0]
        else:
            act["move"] = list(self.nav.local(pos[:2], act["move"]))
        if not loot and (urgent or not evading) and gap > 4:
            trapped = False
            if self.progress_sample is None or now-self.progress_sample[0]>=1.5:
                if self.progress_sample is not None:
                    trapped = distance(pos,self.progress_sample[1]) < 1.8 and any(act["move"])
                self.progress_sample = (now, tuple(pos[:2]))
            toward_zone = direction(pos,goal)
            ahead = (pos[0]+10*toward_zone[0],pos[1]+10*toward_zone[1])
            needs_lift = self.nav.needs_jet and not self.floor.clear(pos,ahead)
            if ((needs_lift or trapped) and now >= self.next_jet and me.get("grounded",True)
                    and me["energy"] >= 0.45*me["max_energy"]):
                self.jet_until, self.next_jet = now+2.0, now+7.0
                act["jump"], act["boost"] = True, False
            if now < self.jet_until and not me.get("grounded",True):
                act["move"] = list(direction(pos,goal))
                act["boost"] = me["energy"] > 0.20*me["max_energy"]
        else:
            self.progress_sample = None
        # In recovery keep Evade's moving orbit in an open safe disk. An uncontested
        # drop or a free cube can still restore HP, and zone urgency takes precedence.
        if self.state=="RECOVER" and not urgent and not loot and recover_move:
            act["move"] = list(self.nav.local(pos[:2], recover_move))
        mechs = [m for m in obs.get("mechs", ()) if m.get("kind")=="mech"]
        low = [m for m in mechs if m.get("hp") is not None and
               m["hp"] <= 0.34*m.get("max_hp",1400) and distance(pos,m["pos"])<=min(110,self.shoot_r)]
        if low:
            low.sort(key=lambda m: (m["hp"],distance(pos,m["pos"])))
            low = [m for m in low if not self.floor.mask or self.floor.clear(pos,m["pos"])]
        nearby = [m for m in mechs if distance(pos,m["pos"]) <= min(self.shoot_r, 180)]
        laser = next((w for w in me["weapons"] if w and w["kind"]=="laser"),None)
        duel = False
        if len(nearby) == 1 and not urgent and hp >= 0.63:
            other = nearby[0]
            facing = abs(math.remainder(yaw_to(other["pos"],pos)-other.get("aim",[0])[0],2*math.pi))
            duel = (other.get("hp", me["hp"]) <= 1.2*me["hp"] and
                    (bool(act.get("fire")) or facing < 0.60 or other["id"] in self.attackers
                     or (laser and hp>0.80 and distance(pos,other["pos"])>50)) and
                    (not self.floor.mask or self.floor.clear(pos,other["pos"])))
        drones = [m for m in obs.get("mechs", ()) if m.get("kind") == "npc"
                  and distance(pos,m["pos"]) <= 48]
        target, item_kind = None, None
        if low and (not urgent or hp > 0.5):
            target, item_kind = low[0], "mech"
        elif duel:
            target, item_kind = nearby[0], "mech"
        elif act.get("fire") and mechs:
            # House bots never damage each other. Preserve progress on one game HP
            # bar when several opponents move through the same engagement.
            eligible = [m for m in nearby if m["id"] == self.focus or m["id"] in self.attackers or
                        distance(pos,m["pos"]) < self.danger_r or
                        abs(math.remainder(yaw_to(m["pos"],pos)-m.get("aim",[0])[0],2*math.pi)) < 0.60]
            if eligible:
                def remaining(m):
                    return (m.get("hp",1400)/max(1,m.get("max_hp",1400))
                            + 0.0015*distance(pos,m["pos"])-0.22*(m["id"]==self.focus))
                target = min(eligible,key=remaining)
            else:
                def yaw_error(m):
                    return abs(math.remainder(yaw_to(pos,m["pos"])-act["aim"][0],2*math.pi))
                target = min(mechs,key=yaw_error)
            item_kind = "mech"
        elif drones and not urgent and hp >= 0.62 and not mechs:
            target, item_kind = min(drones, key=lambda m: distance(pos,m["pos"])), "npc"
        elif loot and loot[0][0]=="crates" and loot[2]:
            target, item_kind = loot[1], "crate"
        if target:
            point = target["pos"]
            d = distance(pos, point)
            muzzle = me.get("stats",{}).get("muzzle_height",7)
            if item_kind == "crate":
                height = self.cfg.get("crates",{}).get("height",2.4)*0.5
            elif item_kind == "npc":
                height = self.cfg.get("npc",{}).get("height",3.5)*0.5
            else:
                height = 4.5
            act["aim"] = [yaw_to(pos, point), math.atan2(point[2]+height-pos[2]-muzzle, max(0.1,d))]
            if item_kind == "mech":
                self.focus = target["id"]
            close_count = sum(distance(pos,m["pos"]) < 65 for m in nearby)
            # The charged game beam costs movement. When a single rival is
            # engaging, preserve the ability to finish a charge even at low HP.
            need_speed = urgent or (evading and not duel and close_count>=2 and hp<0.40)
            self.game_triggers(obs, act, d, bool(low and target is low[0]), need_speed)
            if item_kind=="crate":
                # Stop short enough to keep the crate in the permitted pitch range.
                # The loose cube becomes the next walking goal after the crate breaks.
                if d < 12:
                    if d < 8:
                        act["move"] = list(self.nav.local(pos[:2], direction(point, pos)))
                    else:
                        act["move"] = [0.0,0.0]
                    act["boost"] = False
        else:
            act.pop("fire",None)
            gun = next((w for w in me["weapons"] if w and w["kind"]=="gun"), None)
            if gun and gun.get("reload",0)==0 and gun.get("ammo",12) <= 0.5*gun.get("stats",{}).get("mag",12):
                act["reload"] = True
        if not act.get("fire") and self.state=="EVADE" and self.threat_bearings:
            act["aim"] = [self.threat_bearings[-1][1], 0.0]
        if "aim" in act:
            pitch = act["aim"][1]
            stats = me.get("stats", {})
            act["aim"] = [math.remainder(act["aim"][0],2*math.pi),
                          max(stats.get("min_pitch",-0.7),min(stats.get("max_pitch",1.3),pitch))]
        mf.draw.clear()
        mf.draw.point([goal[0],goal[1],pos[2]], "#40c0ff", 6)
        mf.draw.label(f"{self.state} {self.objective[0] if self.objective else 'zone'}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
