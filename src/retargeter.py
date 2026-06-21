import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

class SimpleRetargeter:
    def __init__(self, hand: OrcaHand):
        self.hand = hand

    def bone_angle_deg(self, a, b, c) -> float:
        """Angle at joint b between bones a→b and b→c."""
        v1 = a - b
        v2 = c - b
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    def palm_frame(self, lm):
        """Returns (forward, sideways, normal) unit vectors of the palm plane."""
        wrist     = lm[0]
        index_mcp = lm[5]
        pinky_mcp = lm[17]

        # Normal points out of the palm face (for right hand)
        radial   = index_mcp - wrist
        ulnar    = pinky_mcp - wrist
        normal   = np.cross(radial, ulnar)
        normal  /= np.linalg.norm(normal)

        # Sideways = knuckle line from pinky to index
        sideways  = index_mcp - pinky_mcp
        sideways /= np.linalg.norm(sideways)

        # Forward = out from palm, perpendicular to knuckle line
        forward   = np.cross(sideways, normal)
        forward  /= np.linalg.norm(forward)

        return forward, sideways, normal

    def abduction_deg(self, mcp, pip, forward, sideways, normal) -> float:
        """
        Lateral angle of finger relative to palm forward direction.
        Positive = spreading outward (abduction), negative = inward (adduction).
        """
        finger_vec = pip - mcp
        # Project onto palm plane (remove normal component)
        in_plane = finger_vec - np.dot(finger_vec, normal) * normal
        in_plane /= np.linalg.norm(in_plane) + 1e-8

        # Lateral component along knuckle line
        lateral = np.dot(in_plane, sideways)
        return np.degrees(np.arcsin(np.clip(lateral, -1.0, 1.0)))

    def landmarks_to_orca(self, lm: np.ndarray) -> OrcaJointPositions:
        """
        lm: (21, 3) MediaPipe world landmarks, wrist at lm[0].
        """
        lm = lm - lm[0]  # normalize: wrist to origin
        roms = self.hand.config.joint_roms_dict

        def clamp(val, joint):
            lo, hi = roms[joint]
            return float(np.clip(val, lo, hi))

        forward, sideways, normal = self.palm_frame(lm)

        angles = {
            # Wrist — pass through as 0 for now, or wire up separately
            "wrist": clamp(0.0, "wrist"),

            # Thumb
            "thumb_cmc": clamp(180 - self.bone_angle_deg(lm[0],  lm[1], lm[2]),  "thumb_cmc"),
            "thumb_abd": clamp(self.abduction_deg(lm[1], lm[2], forward, sideways, normal), "thumb_abd"),
            "thumb_mcp": clamp(180 - self.bone_angle_deg(lm[1],  lm[2], lm[3]),  "thumb_mcp"),
            "thumb_dip": clamp(180 - self.bone_angle_deg(lm[2],  lm[3], lm[4]),  "thumb_dip"),

            # Index
            "index_abd": clamp(self.abduction_deg(lm[5],  lm[6],  forward, sideways, normal), "index_abd"),
            "index_mcp": clamp(180 - self.bone_angle_deg(lm[0],  lm[5],  lm[6]),  "index_mcp"),
            "index_pip": clamp(180 - self.bone_angle_deg(lm[5],  lm[6],  lm[7]),  "index_pip"),

            # Middle
            "middle_abd": clamp(self.abduction_deg(lm[9],  lm[10], forward, sideways, normal), "middle_abd"),
            "middle_mcp": clamp(180 - self.bone_angle_deg(lm[0],  lm[9],  lm[10]), "middle_mcp"),
            "middle_pip": clamp(180 - self.bone_angle_deg(lm[9],  lm[10], lm[11]), "middle_pip"),

            # Ring
            "ring_abd":  clamp(self.abduction_deg(lm[13], lm[14], forward, sideways, normal), "ring_abd"),
            "ring_mcp":  clamp(180 - self.bone_angle_deg(lm[0],  lm[13], lm[14]), "ring_mcp"),
            "ring_pip":  clamp(180 - self.bone_angle_deg(lm[13], lm[14], lm[15]), "ring_pip"),

            # Pinky
            "pinky_abd": clamp(self.abduction_deg(lm[17], lm[18], forward, sideways, normal), "pinky_abd"),
            "pinky_mcp": clamp(180 - self.bone_angle_deg(lm[0],  lm[17], lm[18]), "pinky_mcp"),
            "pinky_pip": clamp(180 - self.bone_angle_deg(lm[17], lm[18], lm[19]), "pinky_pip"),
        }

        return OrcaJointPositions(angles)