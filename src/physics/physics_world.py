"""
physics_world.py: Zero-Gravity PyBullet Rigid Body Simulation for Zero-G Lens.

Architecture:
  - Runs PyBullet in DIRECT mode (headless) — no GUI window.
  - All rendering is handled by ARRenderer using the physics state.
  - Each AI-detected object is mapped to a PyBullet rigid body (sphere proxy).
  - Hand gestures are translated to physics forces (attraction, repulsion, freeze).
  - Gravity is configurable at runtime (default: 0.0 = zero-G).

Physics constants:
  - Linear damping: 0.01 (near-zero drag for realistic vacuum feel)
  - Angular damping: 0.02 (slight rotational damping)
  - Timestep: 1/60 s (60 Hz physics tick)
  - Max velocity: clamped to 3.0 m/s to prevent runaway objects
"""
import pybullet as pb
import pybullet_data
import numpy as np
from typing import Dict, List, Any


class PhysicsWorld:
    TIMESTEP    = 1.0 / 60.0
    MAX_VEL     = 3.0            # Maximum linear velocity (m/s)
    BODY_RADIUS = 0.04           # Collision sphere radius (m)
    BODY_MASS   = 0.5            # Mass per physics body (kg)

    def __init__(self, gravity: float = 0.0):
        self.gravity  = gravity
        self.client   = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0, self.gravity, 0, physicsClientId=self.client)
        pb.setTimeStep(self.TIMESTEP, physicsClientId=self.client)

        self.bodies: Dict[int, int]  = {}   # class_id → pybullet body id
        self.frozen: bool            = False
        self._state: Dict[int, dict] = {}

    # ── Object Synchronization ──────────────────────────────────────
    def sync_objects(self, scene_objects: List[Dict[str, Any]]):
        """
        Maintain physics bodies matching the AI-detected scene objects.
        Uses soft blending (70% physics / 30% detection) to prevent jitter.
        """
        current_ids = {obj["class_id"] for obj in scene_objects}

        # Remove bodies for objects that disappeared from the scene
        for cid in list(self.bodies.keys()):
            if cid not in current_ids:
                pb.removeBody(self.bodies[cid], physicsClientId=self.client)
                del self.bodies[cid]
                self._state.pop(cid, None)

        for obj in scene_objects:
            cid      = obj["class_id"]
            tgt_pos  = list(obj["world_pos"])  # [x, y, z]

            if cid not in self.bodies:
                # Create new rigid body at detected position
                shape = pb.createCollisionShape(
                    pb.GEOM_SPHERE, radius=self.BODY_RADIUS,
                    physicsClientId=self.client
                )
                body = pb.createMultiBody(
                    baseMass=self.BODY_MASS,
                    baseCollisionShapeIndex=shape,
                    basePosition=tgt_pos,
                    physicsClientId=self.client,
                )
                pb.changeDynamics(
                    body, -1,
                    linearDamping=0.01,
                    angularDamping=0.02,
                    physicsClientId=self.client,
                )
                self.bodies[cid] = body
            else:
                # Blend physics position toward detection (soft anchor)
                body = self.bodies[cid]
                cur_pos, cur_orn = pb.getBasePositionAndOrientation(
                    body, physicsClientId=self.client
                )
                blended = [cur_pos[i] * 0.70 + tgt_pos[i] * 0.30 for i in range(3)]
                pb.resetBasePositionAndOrientation(
                    body, blended, cur_orn, physicsClientId=self.client
                )

    # ── Gesture → Force Mapping ─────────────────────────────────────
    def apply_hand_forces(self, hand_data: Dict[str, Any]):
        """
        Translate classified hand gestures into PyBullet external forces.

        Gesture → Action mapping:
          PINCH      → Attract nearest object to hand position
          OPEN_PALM  → Repel all objects within 0.5m radius
          FIST       → Freeze all objects (velocity = 0)
          PEACE      → Apply gentle swirl/rotation to nearby objects
          POINT      → Highlight/select object (no force)
        """
        if self.frozen or not hand_data or not hand_data.get("hands"):
            return

        for hand in hand_data["hands"]:
            gesture  = hand.get("gesture", "UNKNOWN")
            wrist    = hand.get("wrist_pos_normalized", (0.5, 0.5))
            wx = wrist[0] * 2.0 - 1.0
            wy = -(wrist[1] * 2.0 - 1.0)
            hand_pos = np.array([wx, wy, 0.5])

            if gesture == "FIST":
                self._freeze_all()
                continue

            for cid, body in self.bodies.items():
                body_pos = np.array(
                    pb.getBasePositionAndOrientation(body, physicsClientId=self.client)[0]
                )
                direction = body_pos - hand_pos
                dist      = float(np.linalg.norm(direction)) + 1e-6

                if gesture == "PINCH" and dist < 0.35:
                    # Attraction: pull toward hand with inverse-square falloff
                    force_mag = min(6.0, 0.5 / (dist ** 2))
                    force = (-direction / dist) * force_mag
                    pb.applyExternalForce(
                        body, -1, force.tolist(), [0, 0, 0],
                        pb.WORLD_FRAME, physicsClientId=self.client
                    )

                elif gesture == "OPEN_PALM" and dist < 0.55:
                    # Repulsion: push away with linear falloff
                    force_mag = max(0.0, (0.55 - dist) / 0.55) * 10.0
                    force = (direction / dist) * force_mag
                    pb.applyExternalForce(
                        body, -1, force.tolist(), [0, 0, 0],
                        pb.WORLD_FRAME, physicsClientId=self.client
                    )

                elif gesture == "PEACE" and dist < 0.4:
                    # Swirl: apply tangential force (cross product with up axis)
                    up    = np.array([0, 0, 1])
                    swirl = np.cross(direction / dist, up) * 3.0
                    pb.applyExternalForce(
                        body, -1, swirl.tolist(), [0, 0, 0],
                        pb.WORLD_FRAME, physicsClientId=self.client
                    )

    def _freeze_all(self):
        """Zero out velocity and angular velocity for all bodies."""
        for body in self.bodies.values():
            pb.resetBaseVelocity(
                body, [0, 0, 0], [0, 0, 0],
                physicsClientId=self.client
            )

    def _clamp_velocities(self):
        """Prevent runaway objects by clamping linear velocity magnitude."""
        for body in self.bodies.values():
            vel, ang = pb.getBaseVelocity(body, physicsClientId=self.client)
            speed = np.linalg.norm(vel)
            if speed > self.MAX_VEL:
                scale    = self.MAX_VEL / speed
                clamped  = [v * scale for v in vel]
                pb.resetBaseVelocity(body, clamped, ang, physicsClientId=self.client)

    # ── Simulation Step ─────────────────────────────────────────────
    def step(self):
        """Advance physics simulation by one timestep and cache state."""
        self._clamp_velocities()
        pb.stepSimulation(physicsClientId=self.client)

        # Update state cache for renderer
        for cid, body in self.bodies.items():
            pos, orn = pb.getBasePositionAndOrientation(body, physicsClientId=self.client)
            vel, ang = pb.getBaseVelocity(body, physicsClientId=self.client)
            self._state[cid] = {
                "pos": tuple(pos),
                "orn": tuple(orn),
                "vel": tuple(vel),
                "ang": tuple(ang),
            }

    def get_state(self) -> Dict[int, dict]:
        return dict(self._state)

    def set_gravity(self, gx: float, gy: float, gz: float):
        pb.setGravity(gx, gy, gz, physicsClientId=self.client)

    def shutdown(self):
        if self.client >= 0:
            pb.disconnect(self.client)
            print("[PHYSICS] PyBullet disconnected.")
