"""Capture-pose bookkeeping independent of CARLA and of neighbor slot order."""
import math
import torch


class OnlineContextBuilder:
    """Keep actual observed frame poses; never invent a missing capture pose."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_id = None
        self.now_tick = None
        self.poses = {}

    def build(self, episode_id, now_tick, ego_inverse, world_poses, slots, ego_speed_mps,
              ttc_s=None):
        if int(now_tick) != now_tick or now_tick < 0:
            raise ValueError('World snapshot frame must be a nonnegative integer')
        now_tick = int(now_tick)
        if episode_id != self.episode_id:
            self.reset()
            self.episode_id = episode_id
        if self.now_tick is not None and now_tick < self.now_tick:
            raise ValueError('Frame moved backwards without an episode reset')
        self.now_tick = now_tick
        if not math.isfinite(float(ego_speed_mps)) or ego_speed_mps < 0:
            raise ValueError('ego_speed_mps must be a finite nonnegative physical speed')
        frame_poses = {}
        for sender, value in world_poses.items():
            matrix = torch.as_tensor(value, dtype=torch.float32)
            if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
                raise ValueError('Each observed world pose must be finite 4x4')
            frame_poses[str(sender)] = matrix.detach().clone()
        self.poses[now_tick] = frame_poses
        selected = []
        for slot in slots:
            sender = slot['sender_id']
            if sender is None:
                selected.append({'sender_id': None})
                continue
            frame = slot['sensor_frame']
            if int(frame) != frame or frame < 0 or frame > now_tick:
                raise ValueError('LiDAR sensor frame is invalid or in the future')
            frame = int(frame)
            capture_pose = self.poses.get(frame, {}).get(str(sender))
            selected.append({'sender_id': sender, 'sensor_frame': frame,
                             'capture_tick': frame, 'sender_world': capture_pose,
                             'capture_pose_available': capture_pose is not None})
        result = {'episode_id': episode_id, 'now_tick': now_tick,
                  'ego_inverse': ego_inverse, 'slots': selected,
                  'ego_speed_mps': float(ego_speed_mps)}
        if ttc_s is not None:
            if len(ttc_s) != len(slots):
                raise ValueError('TTC entries must follow the same neighbor slots')
            result['ttc_s'] = ttc_s
        return result
