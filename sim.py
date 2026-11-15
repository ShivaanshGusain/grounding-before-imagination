"""Batched 2D pushing simulator, renderer, text descriptions and outcome labels.

Everything is a torch tensor with a leading batch dimension, so it runs on GPU
(GX10) or CPU (smoke tests).

Scene: an arena [0,1]^2 with walls, one pusher (index 0) and K pucks (indices 1..K).
- Puck 1 is always RED (the "target" that the outcome questions ask about).
- Each puck is wood (mass 1) or metal (mass 4); metal is drawn darker.
- The table surface is smooth (low friction) or rough (high friction); the surface
  is visible as the background colour.
Actions (discrete, 9): 8 compass directions + stay. The pusher moves kinematically.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- constants
K_PUCKS = 3
N_BODIES = K_PUCKS + 1           # pusher + pucks
R_PUSHER = 0.06
R_PUCK = 0.07
PUSHER_MASS = 2.0
MASS = {"wood": 1.0, "metal": 4.0}
FRICTION = {"smooth": 1.0, "rough": 4.0}   # velocity decay rate per unit time
PUSHER_SPEED = 0.30              # arena units per macro step
SUBSTEPS = 10
E_WALL = 0.8
E_BALL = 0.9

N_ACTIONS = 9
ACTION_NAMES = ["up", "up-right", "right", "down-right", "down",
                "down-left", "left", "up-left", "stay"]
_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)]
ACTION_DIRS = torch.tensor(
    [(dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)) if (dx or dy) else (0.0, 0.0)
     for dx, dy in _DIRS], dtype=torch.float32)

PALETTE = {
    "red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 60),
    "yellow": (220, 200, 40), "purple": (150, 60, 180),
}
OTHER_COLOURS = ["blue", "green", "yellow", "purple"]
BG = {"smooth": (200, 200, 210), "rough": (150, 120, 90)}
PUSHER_RGB = (30, 30, 30)


@dataclass
class Scene:
    """Initial state of a batch of scenes (all bodies at rest)."""
    pos: torch.Tensor        # (B, N, 2)
    mass: torch.Tensor       # (B, N)   pusher mass at index 0
    friction: torch.Tensor   # (B,)
    rgb: torch.Tensor        # (B, N, 3) float in [0,1]
    bg: torch.Tensor         # (B, 3)    float in [0,1]
    meta: list               # per-scene dicts used for text: colours, materials, surface

    def to(self, device):
        return Scene(self.pos.to(device), self.mass.to(device), self.friction.to(device),
                     self.rgb.to(device), self.bg.to(device), self.meta)


def radii(device) -> torch.Tensor:
    return torch.tensor([R_PUSHER] + [R_PUCK] * K_PUCKS, device=device)


# ---------------------------------------------------------------- scene sampling
def sample_scenes(B: int, rng: random.Random, device="cpu") -> Scene:
    rad = radii("cpu")
    margin = 0.02
    pos = torch.empty(B, N_BODIES, 2)
    for b in range(B):
        placed = []
        for i in range(N_BODIES):
            r = rad[i].item()
            for _ in range(1000):
                p = (rng.uniform(r + margin, 1 - r - margin), rng.uniform(r + margin, 1 - r - margin))
                if all(math.dist(p, q) > r + rq + 0.02 for q, rq in placed):
                    break
            placed.append((p, r))
            pos[b, i] = torch.tensor(p)
    mass = torch.empty(B, N_BODIES)
    friction = torch.empty(B)
    rgb = torch.empty(B, N_BODIES, 3)
    bg = torch.empty(B, 3)
    meta = []
    for b in range(B):
        surface = rng.choice(["smooth", "rough"])
        colours = ["red"] + rng.sample(OTHER_COLOURS, K_PUCKS - 1)
        materials = [rng.choice(["wood", "metal"]) for _ in range(K_PUCKS)]
        mass[b, 0] = PUSHER_MASS
        rgb[b, 0] = torch.tensor(PUSHER_RGB) / 255.0
        for k in range(K_PUCKS):
            mass[b, k + 1] = MASS[materials[k]]
            shade = 0.55 if materials[k] == "metal" else 1.0
            rgb[b, k + 1] = torch.tensor(PALETTE[colours[k]]) / 255.0 * shade
        friction[b] = FRICTION[surface]
        bg[b] = torch.tensor(BG[surface]) / 255.0
        meta.append({"surface": surface, "colours": colours, "materials": materials})
    return Scene(pos.to(device), mass.to(device), friction.to(device), rgb.to(device), bg.to(device), meta)


# ---------------------------------------------------------------- physics
def _resolve_pair(pos, vel, mass, i, j, rad, kinematic_i: bool, contact):
    """Elastic collision + overlap correction between body i and j (batched)."""
    d = pos[:, j] - pos[:, i]
    dist = d.norm(dim=-1).clamp_min(1e-6)
    overlap = (rad[i] + rad[j]) - dist
    hit = overlap > 0
    if not hit.any():
        return
    n = d / dist.unsqueeze(-1)
    vi_n = (vel[:, i] * n).sum(-1)
    vj_n = (vel[:, j] * n).sum(-1)
    approaching = (vi_n - vj_n) > 0
    mi, mj = mass[:, i], mass[:, j]
    if kinematic_i:
        # pusher keeps its commanded velocity; puck bounces off it (momentum formula)
        vj_new = ((mj - mi) * vj_n + 2 * mi * vi_n) / (mi + mj)
        upd = hit & approaching
        vel[:, j] += ((vj_new - vj_n) * upd).unsqueeze(-1) * n
        pos[:, j] += (overlap.clamp_min(0) * hit).unsqueeze(-1) * n
    else:
        vi_new = (mi * vi_n + mj * vj_n - mj * E_BALL * (vi_n - vj_n)) / (mi + mj)
        vj_new = (mi * vi_n + mj * vj_n + mi * E_BALL * (vi_n - vj_n)) / (mi + mj)
        upd = hit & approaching
        vel[:, i] += ((vi_new - vi_n) * upd).unsqueeze(-1) * n
        vel[:, j] += ((vj_new - vj_n) * upd).unsqueeze(-1) * n
        corr = (overlap.clamp_min(0) * hit / 2).unsqueeze(-1) * n
        pos[:, i] -= corr
        pos[:, j] += corr
    contact[:, i, j] |= hit
    contact[:, j, i] |= hit


@torch.no_grad()
def simulate(scene: Scene, actions: torch.Tensor):
    """Roll the true physics forward.

    actions: (B, T) long in [0, N_ACTIONS).
    Returns
      states:  (B, T+1, N, 4)  [x, y, vx, vy] after each macro step (index 0 = initial)
      contact: (B, N, N) bool  any contact during the whole episode
    """
    device = scene.pos.device
    B, T = actions.shape
    rad = radii(device)
    dirs = ACTION_DIRS.to(device)
    pos = scene.pos.clone()
    vel = torch.zeros_like(pos)
    contact = torch.zeros(B, N_BODIES, N_BODIES, dtype=torch.bool, device=device)
    dt = 1.0 / SUBSTEPS
    out = [torch.cat([pos, vel], -1)]
    for t in range(T):
        vel[:, 0] = dirs[actions[:, t]] * PUSHER_SPEED
        for _ in range(SUBSTEPS):
            pos[:, 0] += vel[:, 0] * dt
            pos[:, 1:] += vel[:, 1:] * dt
            vel[:, 1:] *= (1.0 - scene.friction * dt).clamp_min(0.0).view(B, 1, 1)
            for j in range(1, N_BODIES):
                _resolve_pair(pos, vel, scene.mass, 0, j, rad, True, contact)
            for i in range(1, N_BODIES):
                for j in range(i + 1, N_BODIES):
                    _resolve_pair(pos, vel, scene.mass, i, j, rad, False, contact)
            # walls
            lo, hi = rad.view(1, -1, 1), 1.0 - rad.view(1, -1, 1)
            below, above = pos < lo, pos > hi
            vel[:, 1:] = torch.where((below | above)[:, 1:], -E_WALL * vel[:, 1:], vel[:, 1:])
            pos = torch.max(torch.min(pos, hi), lo)
        vel[:, 0] = 0.0          # the pusher stops between macro steps
        out.append(torch.cat([pos, vel], -1))
    return torch.stack(out, 1), contact


# ---------------------------------------------------------------- rendering
@torch.no_grad()
def render(scene: Scene, states: torch.Tensor, size: int = 64) -> torch.Tensor:
    """states (B, S, N, 4) -> frames (B, S, 3, size, size) float in [0,1]."""
    B, S, N, _ = states.shape
    device = states.device
    coords = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size
    gx = coords.view(1, 1, 1, 1, size)                  # x along width
    gy = (1.0 - coords).view(1, 1, 1, size, 1)          # y up
    px = states[..., 0].unsqueeze(-1).unsqueeze(-1)     # (B,S,N,1,1)
    py = states[..., 1].unsqueeze(-1).unsqueeze(-1)
    rad = radii(device).view(1, 1, N, 1, 1)
    inside = ((gx - px) ** 2 + (gy - py) ** 2) <= rad ** 2   # (B,S,N,H,W)
    img = scene.bg.view(B, 1, 3, 1, 1).expand(B, S, 3, size, size).clone()
    for i in range(N):  # later bodies drawn on top
        m = inside[:, :, i].unsqueeze(2)               # (B,S,1,H,W)
        colour = scene.rgb[:, i].view(B, 1, 3, 1, 1)
        img = torch.where(m, colour.expand_as(img), img)
    return img


def frame_pairs(frames: torch.Tensor) -> torch.Tensor:
    """(B,S,3,H,W) -> (B,S,6,H,W): each step stacked with its previous frame (velocity cue)."""
    prev = torch.cat([frames[:, :1], frames[:, :-1]], 1)
    return torch.cat([frames, prev], 2)


# ---------------------------------------------------------------- actions
def random_actions(B: int, T: int, gen: torch.Generator, device="cpu", forbid=None) -> torch.Tensor:
    allowed = [a for a in range(N_ACTIONS) if not forbid or a not in forbid]
    idx = torch.randint(len(allowed), (B, T), generator=gen)
    return torch.tensor(allowed)[idx].to(device)


def directed_actions(scene: Scene, T: int, rng: random.Random, device="cpu", forbid=None, noise=0.3):
    """Mostly move the pusher towards a randomly chosen puck (so interactions happen)."""
    B = scene.pos.shape[0]
    out = torch.empty(B, T, dtype=torch.long)
    dirs = ACTION_DIRS[:8]
    allowed = [a for a in range(N_ACTIONS) if not forbid or a not in forbid]
    for b in range(B):
        target = rng.randrange(1, N_BODIES)
        p = scene.pos[b, 0].cpu().clone()
        q = scene.pos[b, target].cpu()
        for t in range(T):
            if rng.random() < noise:
                a = rng.choice(allowed)
            else:
                v = q - p
                scores = dirs @ (v / (v.norm() + 1e-6))
                order = scores.argsort(descending=True).tolist()
                a = next(o for o in order if o in allowed)
            out[b, t] = a
            p = p + ACTION_DIRS[a] * PUSHER_SPEED
    return out.to(device)


# ---------------------------------------------------------------- language
_SURFACE_WORDS = {"smooth": ["smooth", "slippery", "polished"], "rough": ["rough", "sticky", "carpeted"]}
_MATERIAL_WORDS = {"wood": ["wooden", "light wooden", "wood"], "metal": ["metal", "heavy metal", "steel"]}


def _region(x, y):
    col = ["left", "middle", "right"][min(int(x * 3), 2)]
    row = ["bottom", "middle", "top"][min(int(y * 3), 2)]
    if row == "middle" and col == "middle":
        return "center"
    if row == "middle":
        return f"middle-{col}"
    if col == "middle":
        return f"{row}-middle"
    return f"{row}-{col}"


def _loc(x, y, style, rng):
    if style == "exact":
        fmt = rng.choice(["at ({:.2f}, {:.2f})", "at x={:.2f}, y={:.2f}", "at position ({:.2f}, {:.2f})"])
        return fmt.format(x, y)
    return rng.choice(["in the {} area", "near the {} of the table", "in the {} region"]).format(_region(x, y))


def describe(scene: Scene, b: int, rng: random.Random, style: str = "exact") -> str:
    """Scene description ONLY (no actions, no question): the writer's input."""
    m = scene.meta[b]
    pos = scene.pos[b].cpu()
    surf = rng.choice(_SURFACE_WORDS[m["surface"]])
    parts = [rng.choice(["The table is {}.", "A {} table.", "The surface of the table is {}."]).format(surf)]
    parts.append(rng.choice(["The robot pusher is {}.", "A black pusher is {}.", "The pusher starts {}."])
                 .format(_loc(*pos[0].tolist(), style, rng)))
    order = list(range(K_PUCKS))
    rng.shuffle(order)
    for k in order:
        mat = rng.choice(_MATERIAL_WORDS[m["materials"][k]])
        col = m["colours"][k]
        parts.append(rng.choice(["A {c} {m} puck is {l}.", "There is a {c} {m} puck {l}.", "The {c} puck ({m}) is {l}."])
                     .format(c=col, m=mat, l=_loc(*pos[k + 1].tolist(), style, rng)))
    return " ".join(parts)


# ---------------------------------------------------------------- outcome labels
TASKS = {  # name -> number of classes
    "red_contacted": 2,     # did the pusher touch the red puck?
    "red_moved": 2,         # did the red puck end > 0.10 from its start?
    "red_hit_other": 2,     # did the red puck collide with another puck?
    "red_final_region": 9,  # 3x3 cell of the red puck at the end
}


def outcome_labels(states: torch.Tensor, contact: torch.Tensor) -> dict:
    red0 = states[:, 0, 1, :2]
    redT = states[:, -1, 1, :2]
    cell = (redT * 3).long().clamp(0, 2)
    return {
        "red_contacted": contact[:, 0, 1].long(),
        "red_moved": ((redT - red0).norm(dim=-1) > 0.10).long(),
        "red_hit_other": contact[:, 1, 2:].any(-1).long(),
        "red_final_region": (cell[:, 1] * 3 + cell[:, 0]).long(),
    }
