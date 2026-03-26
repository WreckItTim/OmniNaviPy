import numpy as np

# this class is used to build world representations of the global map based on local sensor observations
class World:
    # this is a aerial-view top-down birds-eye-view 2D occupancy grid where each cell is 1 meter by 1 meter and the value of each cell is:
        # -1-- unknown
        # 0 -- free
        # 1 -- occupied
    # top left corner of the grid is (origin_x, origin_y) in global coordinates, and the grid extends to the right (+x) and downwards (+y) from there
    def __init__(self, global_width, global_height, origin_x, origin_y):
        # global occupancy grid at1-meter resolution
        self.global_grid = np.full((global_height, global_width), -1, dtype=np.int8)
        
        # boundaries of the global map
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.global_min_x = origin_x
        self.global_max_x = origin_x + global_width - 1
        self.global_min_y = origin_y
        self.global_max_y = origin_y + global_height - 1

    # input a depth map taken at the given pose with the given camera parameters, and update the global occupancy grid 
    # assumed point coordinate system of global frame are in standard euclidean and euler space as used throughout this repository
    # all angles are input as degrees
    # assumed coordinate system used in calculations wrt camera frame (not to be confused with pose_x and pose_y which are in global frame):
        # +X: Points Right
        # +Y: Points Down (Towards the floor / gravity)
        # +Z: Points Forward (Out of the camera lens)
        # Pitch (Rx): Rotation around the local X-axis. Positive Pitch tilts the camera down towards the floor. Negative Pitch tilts the camera up towards the sky.
        # Roll (Rz): Rotation around the local Z-axis. Positive Roll tilts the camera counter-clockwise (left wing down, right wing up).
        # Yaw (Heading): Rotation applied to the 2D plane. Positive Yaw rotates the robot Counter-Clockwise (turning left). A Negative Yaw turns the robot right.
    def update_map_from_depth(self, depth_map, point, is_planar_depth=True,
                              horizontal_fov=90.0, vertical_fov=90.0, height_range=(-3.0, 3.0), horizon=255.0):

        # convert input coordinates to match the coordinate system used in calculations
        pose_x, pose_y, pose_yaw, pitch, roll = point.x, point.y, point.yaw, point.pitch, point.roll
        pose_y *= -1 # flip y-axis of pose to match the coordinate system used in calculations (where +y is downwards)
        yaw = -np.deg2rad(pose_yaw) - np.pi / 2 # convert, invert, and shift 
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        pitch = -pitch # invert pitch to match the coordinate system used in calculations (where positive pitch is downwards)
        roll = -roll # invert roll to match the coordinate system used in calculations (where positive roll is counter-clockwise)
        h_fov_rad = np.deg2rad(horizontal_fov)
        v_fov_rad = np.deg2rad(vertical_fov)

        # get camera parameters
        height, width = depth_map.shape
        fx = width / (2.0 * np.tan(h_fov_rad / 2.0))
        fy = height / (2.0 * np.tan(v_fov_rad / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        # calculate 3D pixel coordinates in camera frame
        v, u = np.indices((height, width))
        if is_planar_depth:
            z_c = depth_map
        else:
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
            ray_length = np.sqrt(x_norm**2 + y_norm**2 + 1.0)
            z_c = depth_map / ray_length
        x_c_flat = ((u - cx) * z_c / fx).ravel()
        y_c_flat = ((v - cy) * z_c / fy).ravel()
        z_c_flat = z_c.ravel()
        
        # apply rotations due to pitch and roll of the camera
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R_local = Rz @ Rx
        R_local_inv = R_local.T
        P_c = np.vstack((x_c_flat, y_c_flat, z_c_flat))
        P_w = R_local @ P_c
        x_w_flat = P_w[0]
        y_w_flat = P_w[1]
        z_w_flat = P_w[2]
        
        # transform the raw 3D point cloud into global coordinates to check bounds
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        X_global_pts = x_w_flat * cos_y - z_w_flat * sin_y + pose_x
        Y_global_pts = x_w_flat * sin_y + z_w_flat * cos_y + pose_y
        
        ## filter out not needed points 
        # only consider points that are within the horizon distance and in front of the camera
        valid_mask = (z_w_flat < horizon) & (z_w_flat > 0)
        # apply height range to filter out things like the ground and high-hanging telephone wires that we don't want to consider as obstacles
        if height_range is not None:
            y_min, y_max = height_range
            valid_mask &= (y_w_flat >= y_min) & (y_w_flat <= y_max)
        # filter out points that fall out of global map bounds
        valid_mask &= (X_global_pts >= self.global_min_x) & (X_global_pts <= self.global_max_x)
        valid_mask &= (Y_global_pts >= self.global_min_y) & (Y_global_pts <= self.global_max_y)
        valid_u = u.ravel()[valid_mask]
        valid_z_w = z_w_flat[valid_mask]
        
        # now get the closest obstacle depth for each column of pixels in the depth map (if no valid points in a column, then set to infinity)
        obs_z_per_col = np.full(width, np.inf)
        if len(valid_u) > 0:
            np.minimum.at(obs_z_per_col, valid_u, valid_z_w)
            
        # slice out the pixels that correspond to the overlapping area between the robot's pose and global map
        min_x_eval = max(self.global_min_x, pose_x - horizon)
        max_x_eval = min(self.global_max_x, pose_x + horizon)
        min_y_eval = max(self.global_min_y, pose_y - horizon)
        max_y_eval = min(self.global_max_y, pose_y + horizon)
        idx_min_x = int(np.floor(min_x_eval - self.origin_x))
        idx_max_x = int(np.floor(max_x_eval - self.origin_x))
        idx_min_y = int(np.floor(min_y_eval - self.origin_y))
        idx_max_y = int(np.floor(max_y_eval - self.origin_y))
        patch = self.global_grid[idx_min_y:idx_max_y+1, idx_min_x:idx_max_x+1]
        
        # get pixel indicies based on center of pixel (add 0.5)
        idx_x_arr = np.arange(idx_min_x, idx_max_x + 1)
        idx_y_arr = np.arange(idx_min_y, idx_max_y + 1)
        IDX_X, IDX_Y = np.meshgrid(idx_x_arr, idx_y_arr)
        X_grid_global = IDX_X + self.origin_x + 0.5
        Y_grid_global = IDX_Y + self.origin_y + 0.5
        
        # 5. Inverse Translation and Yaw (Global -> Local Base Frame)
        dx = X_grid_global - pose_x
        dy = Y_grid_global - pose_y
        
        X_grid_local = dx * cos_y + dy * sin_y
        Z_grid_local = -dx * sin_y + dy * cos_y
        
        # 6. Inverse Pitch/Roll (Local Base -> Camera Frame)
        P_grid_local = np.vstack((X_grid_local.ravel(), np.zeros_like(X_grid_local.ravel()), Z_grid_local.ravel()))
        P_grid_cam = R_local_inv @ P_grid_local
        
        X_grid_c = P_grid_cam[0].reshape(X_grid_local.shape)
        Z_grid_c = P_grid_cam[2].reshape(X_grid_local.shape)
        
        u_grid = np.round((X_grid_c * fx / (Z_grid_c + 1e-6)) + cx).astype(np.int32)
        
        # 7. Apply FOV constraint to the global slice
        fov_mask = (u_grid >= 0) & (u_grid < width) & (Z_grid_c > 0) & (Z_grid_local > 0)
        
        Z_local_in_fov = Z_grid_local[fov_mask]
        obs_z_in_fov = obs_z_per_col[u_grid[fov_mask]]

        # 8. Evaluate Occupancy
        free_mask = Z_local_in_fov < (obs_z_in_fov - 1.0)
        occ_mask = (np.abs(Z_local_in_fov - obs_z_in_fov) <= 1.5) & (obs_z_in_fov != np.inf)

        # 9. Modify the extracted slice directly
        flat_patch = patch[fov_mask]
        flat_patch[free_mask] = 0
        flat_patch[occ_mask] = 1
        patch[fov_mask] = flat_patch
        
        # 10. Plug the slice right back into the global map!
        self.global_grid[idx_min_y:idx_max_y+1, idx_min_x:idx_max_x+1] = patch
        
        return self.global_grid



    # # is_planar_depth is an option to treat depth values as planar Z values instead of radial/Euclidean distances
    # # makes grid where -1 denotes unknown space, 0 denotes free space, and 1 denotes occupied space
    # # point is the location of the camera in the world frame
    # # roll and pitch are in radians
    # # outputs 1 meter resolution grid and the min/max x and z values of the grid in meters
    # def get_relative_occupancy_grid(self, depth_map, point, height_range=(-3, 3), horizon=255, is_planar_depth=False,
    #                         horizontal_fov=90, vertical_fov=90, roll=0, pitch=0):

    #     # X axis points right, Y axis points upward, Z axis points forward in camera frame

    #     # set intrinsic camera parameters
    #     height, width = depth_map.shape

    #     # calulate focal length based on horizontal and vertical FOV, and image dimensions
    #     fx = width / (2.0 * np.tan(np.deg2rad(horizontal_fov) / 2.0))
    #     fy = height / (2.0 * np.tan(np.deg2rad(vertical_fov) / 2.0))
        
    #     # assume principal point is at center of image
    #     cx = width / 2.0
    #     cy = height / 2.0 

    #     # calculate 3D pixel coordinates
    #     v, u = np.indices((height, width))
    #     x_norm = (u - cx) / fx
    #     y_norm = (v - cy) / fy
    #     if is_planar_depth:
    #         z = depth_map
    #     else:
    #         z = depth_map / np.sqrt(x_norm**2 + y_norm**2 + 1.0)
    #     x, y = x_norm * z, y_norm * z # multiply by depth to get 3D coordinates in camera frame
        
    #     ## rotate pixels to adjust for pitch and roll
    #     # pitch (around X axis)
    #     Rx = np.array([
    #         [1, 0, 0],
    #         [0, np.cos(pitch), -np.sin(pitch)],
    #         [0, np.sin(pitch), np.cos(pitch)]
    #     ])
    #     # roll (around Z axis)
    #     Rz = np.array([
    #         [np.cos(roll), -np.sin(roll), 0],
    #         [np.sin(roll), np.cos(roll), 0],
    #         [0, 0, 1]
    #     ])
    #     # combined rotation matrix
    #     R = np.matmul(Rz, Rx)
    #     # rotate x, y, z coordinates to global orientation frame -- returns flattened
    #     x, y, z = np.matmul(R, np.vstack((x.ravel(), y.ravel(), z.ravel())))

    #     # only get depths less than horizon and greater than 0
    #     mask = (z < horizon) & (z > 0)
    #     # filter out points outside of height range (like the ground and high-hanging telephone wires)
    #     if height_range is not None:
    #         y_min, y_max = height_range
    #         mask &= (y >= y_min) & (y <= y_max)
    #     # apply mask to only consider pixels that meet the above criteria
    #     u, z = u.ravel()[mask], z[mask]

    #     # we want to find the closest object at each u-index column
    #     obstacles = np.full(width, np.inf)
    #     if len(u) > 0:
    #         np.minimum.at(obstacles, u, z)
        
    #     # make XZ occupancy grid where -1 denotes unknown space, 0 denotes free space, and 1 denotes occupied space
    #     x_min, x_max = -horizon, horizon 
    #     z_min, z_max = 0, horizon
    #     grid_width = int(np.ceil((x_max - x_min)))
    #     grid_height = int(np.ceil((z_max - z_min)))
    #     occupancy_grid = np.full((grid_height, grid_width), -1, dtype=np.int8)
        
    #     # make grid of x and z coordinates where each pixel is cenetered at the middle of the pixel
    #     x_coords = np.linspace(x_min + 1/2, x_max - 1/2, grid_width)
    #     z_coords = np.linspace(z_min + 1/2, z_max - 1/2, grid_height)
    #     X_grid, Z_grid = np.meshgrid(x_coords, z_coords)

    #     # rotate back to camera frame
    #     X_grid, _, Z_grid = np.matmul(R.T, np.vstack((X_grid.ravel(), np.zeros_like(X_grid.ravel()), Z_grid.ravel()))).reshape(3, grid_height, grid_width)
    
    #     # translate rotated x-values to indicies
    #     u_grid = np.round((X_grid * fx / (Z_grid + 1e-6)) + cx).astype(np.int32)
        
    #     # only consider pixels in our field of view defined by x_min and x_max
    #     mask = (u_grid >= 0) & (u_grid < width)
    #     u_grid = u_grid[mask]
    #     Z_grid = Z_grid[mask]
    #     obstacles = obstacles[u_grid]

    #     # free space is where the depth value is less than the obstacle depth
    #     free_space = Z_grid < (obstacles - 1)

    #     # occupied space is padded around the obstacle depth to account for noise
    #     occupied_space = (np.abs(Z_grid - obstacles) <= (1.5)) & (obstacles != np.inf)
        
    #     # update occupancy grid values based on free and occupied space
    #     flat_grid = occupancy_grid[mask]
    #     flat_grid[free_space] = 0
    #     flat_grid[occupied_space] = 1
    #     occupancy_grid[mask] = flat_grid
        
    #     return occupancy_grid, (x_min, x_max, z_min, z_max)
        
    # def update(self, depth_map, point):
    #     x, y = point.x, point.y
    #     if x < self.xmin or x >= self.xmax or y < self.ymin or y >= self.ymax:
    #         return
    #     relative_occupancy_grid, bounds = self.get_relative_occupancy_grid(depth_map, point)
    #     horizon = 255
    #     if point.direction == 0:
    #         xmin = point.x - horizon
    #         xmax = point.x + horizon
    #         ymin = point.y
    #         ymax = point.y + horizon 
    #     if point.direction == 1:
    #         relative_occupancy_grid = np.rot90(relative_occupancy_grid, k=1)
    #         xmin = point.x
    #         xmax = point.x + horizon
    #         ymin = point.y - horizon
    #         ymax = point.y + horizon
    #     if point.direction == 2:
    #         relative_occupancy_grid = np.rot90(relative_occupancy_grid, k=2)
    #         xmin = point.x - horizon
    #         xmax = point.x + horizon
    #         ymin = point.y - horizon 
    #         ymax = point.y
    #     if point.direction == 3:
    #         relative_occupancy_grid = np.rot90(relative_occupancy_grid, k=3)
    #         xmin = point.x - horizon
    #         xmax = point.x
    #         ymin = point.y - horizon
    #         ymax = point.y + horizon
    #     self.update_grid(relative_occupancy_grid.T, xmin, xmax, ymin, ymax)

    # def update_grid(self, occ_grid, xmin, xmax, ymin, ymax):
    #     # bound inside of world grid
    #     xmin2 = max(self.xmin, xmin)
    #     xmax2 = min(self.xmax, xmax)
    #     ymin2 = max(self.ymin, ymin)
    #     ymax2 = min(self.ymax, ymax)
    #     occ_grid = occ_grid[xmin2-xmin:xmax2-xmin, ymin2-ymin:ymax2-ymin]
    #     world_grid = self.grid[xmin2-self.xmin:xmax2-self.xmin, ymin2-self.ymin:ymax2-self.ymin]
    #     flip_mask = (occ_grid>-1)
    #     world_grid[flip_mask] = occ_grid[flip_mask]
    #     self.grid[xmin2-self.xmin:xmax2-self.xmin, ymin2-self.ymin:ymax2-self.ymin] = world_grid