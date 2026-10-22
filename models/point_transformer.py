import torch
import torch.nn as nn
import random
import open3d.ml.torch as ml3d
from .pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from .transformer_block import TransformerBlock
from cooperative.config import TemporalConfig
from cooperative.messages import CausalMessageBuffer
from cooperative.rr_gate import risk_from_ttc, compute_gate, apply_log_gate
from cooperative.tarc import TARC
from cooperative.pooling import zero_floor_backward


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        
simple=False
class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, config.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        if not simple:
            for i in range(nblocks):
                channel = 32 * 2 ** (i + 1)
                self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
                self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))
        else:
            for i in range(nblocks):
                channel = 32
                self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel + 3, channel, channel]))
                self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))

        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            #from IPython import embed; embed()
            #xyz_and_feats.append((xyz, points))
        return points, xyz, xyz_and_feats

class PointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone(config)
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks+8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
        dim_meta = 1
        self.fc_meta = nn.Sequential(
            nn.Linear(dim_meta, 16),
            nn.ReLU(),
            nn.Linear(16,8)
            )
    def forward(self, ego_lidar, ego_speed):
        points, xyz, _ = self.backbone(ego_lidar)
        ego_meta = self.fc_meta(ego_speed[:,None])
        #points = points.mean(1)
        points = points.max(1).values
        concat_output = torch.cat((points, ego_meta),dim=-1)
        pred_control = self.fc2(concat_output)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer

class CooperativePointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone(config)
        self.backbone_other = Backbone(config)
        #self.backbone_other.register_backward_hook(hook_fn)
        npoints, nblocks, nneighbor, n_c, d_points = config.npoints, config.nblocks, config.nneighbor, config.num_output, config.input_dim
        if not simple:
            channel = 32 * 2 ** (nblocks+1)
            self.aggr_transition_down = TransitionDown(npoints // 4 ** (nblocks+1) * 2, nneighbor, [channel //2 +3, channel, channel]) #(1+config.max_num_neighbors),
        else:
            channel = 32
            self.aggr_transition_down = TransitionDown(npoints // 4 ** (nblocks+1) * 2, nneighbor, [channel +3, channel, channel]) #(1+config.max_num_neighbors),
        self.aggr_transformer = TransformerBlock(channel , config.transformer_dim, nneighbor)#channel was  divided  by 2
        if not simple:
            self.fc2 = nn.Sequential(
                nn.Linear(32 * 2 ** (nblocks+1) + 8, 256), # was  divided by 2 , was +8
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_c)
            )
        else:
            self.fc2 = nn.Sequential(
                nn.Linear(channel + 8, 256), # was  divided by 2 , was +8
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_c)
            )

        self.nblocks = nblocks
        self.npoints = npoints
        dim_meta = 1
        self.fc_meta = nn.Sequential(
            nn.Linear(dim_meta, 16),
            nn.ReLU(),
            nn.Linear(16,8)
            )
        self.voxel_pooling = ml3d.layers.VoxelPooling(position_fn='average', feature_fn='max')

        # No new parameters or random draws on the legacy/disabled path.
        self.temporal_config = TemporalConfig.from_config(config)
        self.temporal_aux = []
        self.temporal_trace = []
        self.temporal_publications = None
        self._temporal_episode = None
        if self.temporal_config is not None:
            if npoints != 2048 or nblocks != 2 or d_points != 3 or simple:
                raise ValueError('TARC requires the 2048-point, two-block, 128-feature CPT backbone')
            self.tarc = TARC(self.temporal_config)
            self.message_buffer = CausalMessageBuffer(self.temporal_config.delay_ms)

    def reset_temporal_state(self):
        """Reset inference history at an episode/reconnection boundary."""
        if self.temporal_config is not None:
            self.message_buffer.reset()
        self._temporal_episode = None
        self.temporal_aux = []
        self.temporal_trace = []
        self.temporal_publications = None

    def forward(self, ego_lidar, ego_speed, other_lidar, other_transform,
                temporal_context=None, temporal_batch=None):
        if self.temporal_config is None:
            return self._forward_legacy(ego_lidar, ego_speed, other_lidar, other_transform)
        if (temporal_context is None) == (temporal_batch is None):
            raise ValueError('Enabled TARC needs exactly one of temporal_context or temporal_batch')
        self.temporal_aux = []
        self.temporal_trace = []
        if temporal_batch is not None:
            self.temporal_publications = None
            return self._forward_temporal_batch(ego_lidar, ego_speed, temporal_batch)
        if self.training:
            raise ValueError('Online message caches are inference-only; train with temporal_batch')
        with torch.no_grad():
            return self._forward_temporal_online(ego_lidar, ego_speed, other_lidar, temporal_context)

    @staticmethod
    def _transform_xyz(xyz, transform):
        # Match the existing coordinate transform; never rotate feature channels.
        return torch.transpose(torch.matmul(transform[:, :3, :3],
                                            torch.transpose(xyz, 1, 2)), 1, 2) + torch.cat(
            xyz.shape[1] * [transform[:, :3, 3].unsqueeze(1)], dim=1)

    def _gate_output(self, output, age, speed_mps, ttc):
        if not self.temporal_config.rr_gate:
            return output['features'], None
        risk, ttc_valid = risk_from_ttc(ttc, speed_mps)
        gate = compute_gate(output['quality'], age, risk)
        return apply_log_gate(output['features'], gate), {
            'risk': risk.detach(), 'ttc_valid': ttc_valid.detach(), 'gate': gate.detach()}

    def _forward_temporal_online(self, ego_lidar, ego_speed, other_lidar, context):
        if ego_lidar.size(0) != 1 or other_lidar.size(0) != 1:
            raise ValueError('Online context represents one episode; use temporal_batch for batches')
        slots = context['slots']
        if len(slots) != other_lidar.size(1):
            raise ValueError('Temporal slots must match the original neighbor tensor')
        episode = context['episode_id']
        if episode != self._temporal_episode:
            self.reset_temporal_state()
            self._temporal_episode = episode
        now = context['now_tick']
        self.message_buffer.advance(now)
        points, xyz, _ = self.backbone(ego_lidar)
        ego_inverse = torch.as_tensor(context['ego_inverse'], device=xyz.device, dtype=xyz.dtype)
        if ego_inverse.shape != (4, 4) or not torch.isfinite(ego_inverse).all():
            raise ValueError('ego_inverse must be a finite 4x4 capture-compatible transform')
        speed = torch.as_tensor(context['ego_speed_mps'], device=xyz.device, dtype=xyz.dtype)
        candidates = []
        self.temporal_publications = {
            'schema_version': 1, 'episode_id': str(episode), 'decision_tick': now,
            'slots': [source['sender_id'] for source in slots], 'published': []}
        for slot, source in enumerate(slots):
            sender = source['sender_id']
            if sender is None:
                continue
            if source.get('sensor_frame') is not None and source['sensor_frame'] != source['capture_tick']:
                raise ValueError('Schema 1 uses the same absolute world frame for capture_tick and sensor_frame')
            # A missing capture pose must never be replaced with the current pose.
            if source.get('sender_world') is not None:
                features, positions, _ = self.backbone_other(other_lidar[:, slot])
                pose = torch.as_tensor(source['sender_world'], device=xyz.device, dtype=xyz.dtype)
                published = self.message_buffer.publish(sender, source['capture_tick'], features[0],
                                                        positions[0], pose, source.get('sensor_frame'))
                if published:
                    self.temporal_publications['published'].append({
                        'sender_id': sender, 'capture_tick': source['capture_tick'],
                        'sensor_frame': source['capture_tick'] if source.get('sensor_frame') is None else source['sensor_frame'],
                        'publish_tick': now})
                self.message_buffer.advance(now)
            history = self.message_buffer.consume_history(sender)
            if not history:
                self.temporal_trace.append({'sender_id': sender, 'valid': False})
                continue
            local_features = torch.stack([message.features for message in history])
            aligned_xyz = torch.cat([
                self._transform_xyz(message.xyz.unsqueeze(0),
                                    (ego_inverse @ message.sender_world).unsqueeze(0))
                for message in history], dim=0)
            ages = xyz.new_tensor([(now - message.capture_tick) * 0.05 for message in history])
            valid = torch.ones(len(history), device=xyz.device, dtype=torch.bool)
            output = self.tarc(local_features, aligned_xyz, ages, valid)
            ttc = context.get('ttc_s')
            ttc = None if ttc is None else torch.as_tensor(ttc[slot], device=xyz.device, dtype=xyz.dtype)
            fused_features, gate_trace = self._gate_output(output, ages[-1], speed, ttc)
            candidates.append((output['xyz'], fused_features))
            self.temporal_trace.append({
                'sender_id': sender, 'valid': True,
                'capture_ticks': [message.capture_tick for message in history],
                'age_s': ages[-1].detach(), 'quality': output['quality'].detach(),
                'gate': gate_trace, 'prefusion_xyz': output['xyz'].detach(),
                'compensated_features': output['features'].detach(),
                'prefusion_features': fused_features.detach()})
        return self._fuse_temporal(points, xyz, ego_speed, [candidates])

    def _forward_temporal_batch(self, ego_lidar, ego_speed, batch):
        history_lidar = batch['history_lidar'].to(device=ego_lidar.device, dtype=ego_lidar.dtype)
        transforms = batch['history_transform'].to(device=ego_lidar.device, dtype=ego_lidar.dtype)
        ages = batch['history_age_s'].to(device=ego_lidar.device, dtype=ego_lidar.dtype)
        valid = batch['history_valid'].to(device=ego_lidar.device, dtype=torch.bool)
        speed = batch['ego_speed_mps'].to(device=ego_lidar.device, dtype=ego_lidar.dtype)
        if history_lidar.ndim != 5 or history_lidar.shape[0] != ego_lidar.shape[0]:
            raise ValueError('history_lidar must be [B,M,H,P,3]')
        bsize, neighbors, frames, npoints, channels = history_lidar.shape
        if frames != 4 or npoints != self.npoints or channels != 3:
            raise ValueError('TARC batches require four history positions and original point count')
        if transforms.shape != (bsize, neighbors, frames, 4, 4) or ages.shape != valid.shape or valid.shape != (bsize, neighbors, frames):
            raise ValueError('History geometry, ages and masks must share [B,M,H] indices')
        if speed.shape != (bsize,):
            raise ValueError('ego_speed_mps must be [B], in physical m/s')
        if torch.any(ages[valid] < self.temporal_config.delay_ms / 1000.0):
            raise ValueError('History contains a message not yet delivered at the configured delay')
        points, xyz, _ = self.backbone(ego_lidar)
        all_candidates = []
        for b in range(bsize):
            candidates = []
            for sender in range(neighbors):
                mask = valid[b, sender]
                if not bool(mask.any()):
                    continue
                encoded, positions, _ = self.backbone_other(history_lidar[b, sender, mask])
                aligned = self._transform_xyz(positions, transforms[b, sender, mask])
                selected_ages = ages[b, sender, mask]
                output = self.tarc(encoded, aligned, selected_ages,
                                   torch.ones_like(selected_ages, dtype=torch.bool))
                ttc = batch.get('ttc_s')
                ttc = None if ttc is None else ttc[b, sender].to(ego_lidar.device)
                fused_features, _ = self._gate_output(output, selected_ages[-1], speed[b], ttc)
                candidates.append((output['xyz'], fused_features))
                if self.training and bool(batch['reference_valid'][b, sender]):
                    # Current, undelayed features are supervision only, never inference input.
                    modes = [(module, module.training) for module in self.backbone_other.modules()]
                    devices = [ego_lidar.device.index] if ego_lidar.is_cuda else []
                    try:
                        self.backbone_other.eval()
                        # Reference FPS/BN must not affect later causal messages.
                        with torch.random.fork_rng(devices=devices), torch.no_grad():
                            reference = batch['reference_lidar'][b, sender].to(ego_lidar).unsqueeze(0)
                            ref_features, ref_xyz, _ = self.backbone_other(reference)
                            ref_transform = batch['reference_transform'][b, sender].to(ego_lidar).unsqueeze(0)
                            ref_xyz = self._transform_xyz(ref_xyz, ref_transform)
                    finally:
                        for module, mode in modes:
                            module.training = mode
                    self.temporal_aux.append({'output': output,
                                              'reference_features': ref_features[0],
                                              'reference_xyz': ref_xyz[0]})
            all_candidates.append(candidates)
        return self._fuse_temporal(points, xyz, ego_speed, all_candidates)

    def _fuse_temporal(self, ego_points, ego_xyz, ego_speed, candidates):
        groups = {}
        for b, messages in enumerate(candidates):
            points, xyz = ego_points[b:b+1], ego_xyz[b:b+1]
            if messages:
                remote_xyz = torch.cat([message[0] for message in messages], dim=0)
                remote_points = torch.cat([message[1] for message in messages], dim=0)
                pooled = self.voxel_pooling(remote_xyz.to('cpu'), remote_points.to('cpu'), voxel_size=0.5)
                o_xyz, o_points = pooled.pooled_positions, pooled.pooled_features
                if self.training:
                    # User-approved training-only fix; native forward bits stay unchanged.
                    o_points = zero_floor_backward(o_points)
                mask = ((o_xyz[:, 0] >= -70) & (o_xyz[:, 0] <= 70) &
                        (o_xyz[:, 1] >= -70) & (o_xyz[:, 1] <= 70) &
                        (o_xyz[:, 2] > -2) & (o_xyz[:, 2] <= 0.25))
                o_xyz, o_points = o_xyz[mask], o_points[mask]
                if o_xyz.size(0):
                    desired = xyz.shape[1]
                    if o_xyz.size(0) < desired:
                        indices = list(range(o_xyz.size(0)))
                        indices.extend(random.choices(list(range(o_xyz.size(0))),
                                                     k=desired-o_xyz.size(0)))
                    else:
                        indices = random.sample(range(o_xyz.size(0)), desired)
                    indices = torch.tensor(indices, dtype=torch.long)
                    xyz = torch.cat((xyz, o_xyz[indices].to(ego_xyz).unsqueeze(0)), dim=1)
                    points = torch.cat((points, o_points[indices].to(ego_points).unsqueeze(0)), dim=1)
            # Empty remote/ROI uses the same cooperative head on ego features.
            groups.setdefault(xyz.shape[1], []).append((b, xyz, points))
        rows = [None] * ego_xyz.size(0)
        for members in groups.values():
            xyz = torch.cat([member[1] for member in members], dim=0)
            points = torch.cat([member[2] for member in members], dim=0)
            xyz, points = self.aggr_transition_down(xyz, points)
            points = self.aggr_transformer(xyz, points)[0]
            indices = torch.tensor([member[0] for member in members], device=ego_speed.device)
            meta = self.fc_meta(ego_speed[indices, None])
            predictions = self.fc2(torch.cat((points.max(1).values, meta), dim=-1))
            for i, member in enumerate(members):
                rows[member[0]] = predictions[i:i+1]
        predictions = torch.cat(rows, dim=0)
        return predictions[:, 0], predictions[:, 1], predictions[:, 2]

    def _forward_legacy(self, ego_lidar, ego_speed, other_lidar, other_transform):
        
        points, xyz, _ = self.backbone(ego_lidar)
        other_lidars = torch.split(other_lidar, 1, dim=1)
        other_transforms = torch.split(other_transform, 1, dim=1)
        
        for i in range(len(other_lidars)):
            lidar = torch.squeeze(other_lidars[i], dim=1)
            trans = torch.squeeze(other_transforms[i], dim=1)
            other_points, other_xyz, _ = self.backbone_other(lidar)
            # Transform the xyz from other frame into ego frame
            other_xyz = torch.transpose(torch.matmul(trans[:,:3,:3],torch.transpose(other_xyz, 1, 2)), 1,2)+torch.cat(other_points.shape[1]*[trans[:,:3,3].unsqueeze(1)],dim=1)
            if i==0:
                others_xyz = other_xyz
                others_points = other_points
            else:
                others_xyz = torch.cat((others_xyz, other_xyz), dim=1)
                others_points = torch.cat((others_points, other_points), dim=1)
        #Filter Out points outside x[-70,70] y[-70,70], z[-2, -0.25]
        #mask = (others_xyz[:,:,0]>-70)&(others_xyz[:,:,0]<70)&(others_xyz[:,:,1]>-70)&(others_xyz[:,:,1]<70)&(others_xyz[:,:,2]>-2)&(others_xyz[:,:,2]<-0.25)        
        #from IPython import embed; embed() 
        #others_xyz = others_xyz[mask]
        #others_points = others_points[mask]
        #from IPython import embed; embed()
        for i in range(ego_lidar.size(0)): #Batch dimension
            pooled_voxels = self.voxel_pooling(others_xyz[i].to('cpu'), others_points[i].to('cpu'), voxel_size=0.5)
            o_xyz = pooled_voxels.pooled_positions
            o_points = pooled_voxels.pooled_features
            #Filter Out points outside x[-70,70] y[-70,70], z[-2, -0.25]
            mask = (o_xyz[:,0]>=-70)&(o_xyz[:,0]<=70)&(o_xyz[:,1]>=-70)&(o_xyz[:,1]<=70)&(o_xyz[:,2]>-2)&(o_xyz[:,2]<=0.25)        
            #from IPython import embed; embed()
            o_xyz = o_xyz[mask]
            o_points = o_points[mask]

            desired_npoints = xyz.shape[1]#self.npoints // 4 ** (self.nblocks+1) * 2
            if o_xyz.size(0) < desired_npoints:
                #print(o_xyz)
                indice = list(range(max(o_xyz.size(0),1)))
                indice.extend(list(random.choices(list(range(max(o_xyz.size(0),1))), k=int(desired_npoints-o_xyz.size(0)))))
                #print(indice)
            else:
                indice = random.sample(range(o_xyz.size(0)), desired_npoints)
                #print(indice)
            indice = torch.tensor(indice)
            if len(o_xyz.shape) < 3:
                try:
                    o_xyz = o_xyz[indice].unsqueeze(0)
                    o_points = o_points[indice].unsqueeze(0)
                except:
                    from IPython import embed; embed()
            if i==0:
                subsample_others_points = o_points
                subsample_others_xyz = o_xyz
            else:
                try:
                    subsample_others_points = torch.cat((subsample_others_points,o_points), dim=0)
                    subsample_others_xyz = torch.cat((subsample_others_xyz,o_xyz), dim=0)
                except:
                    from IPython import embed; embed()
                    print(o_points)
                    print("===")
                    print(subsample_others_points)
        subsample_others_points = subsample_others_points.to('cuda')
        subsample_others_xyz = subsample_others_xyz.to('cuda')

        points = torch.cat((points, subsample_others_points), dim=1)
        xyz = torch.cat((xyz, subsample_others_xyz), dim=1)
        
        # How to handle the dynamic number of points?
        xyz, points = self.aggr_transition_down(xyz, points)
        #from IPython import embed; embed()
        points = self.aggr_transformer(xyz, points)[0]
        ego_meta = self.fc_meta(ego_speed[:,None])
        #points = points.mean(1)
        points = points.max(1).values
        concat_output = torch.cat((points, ego_meta),dim=-1)
        pred_control = self.fc2(concat_output)
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        pred_control = self.fc2(points.mean(1))
        pred_throttle = pred_control[:,0]
        pred_brake = pred_control[:,1]
        pred_steer = pred_control[:,2]

        return pred_throttle, pred_brake, pred_steer


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, config.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, config.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)


def hook_fn(m, i, o):
  print(m)
  print("------------Input Grad------------")

  for grad in i:
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in o:  
    try:
      print(grad.shape)
    except AttributeError: 
      print ("None found for Gradient")
  print("\n")    

