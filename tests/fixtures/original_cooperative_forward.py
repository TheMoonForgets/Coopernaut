"""Frozen original CPT class from commit 13bb995d179754a72f7880fbe958a311d868332f.

Extracted verbatim from models/point_transformer.py. Tests inject explicitly
labelled lightweight encoders/pooling; this fixture must stay unchanged when
the production temporal implementation changes.
"""
import random
import torch
import torch.nn as nn
simple = False

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

    def forward(self, ego_lidar, ego_speed, other_lidar, other_transform):
        
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
