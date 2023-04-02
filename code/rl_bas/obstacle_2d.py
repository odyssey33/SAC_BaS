import numpy as np

class Obstacles2D:
    def __init__(self, n, obst_course_type, number_of_obstacles=None):
        self.n = n #number of states (x, y, theta) => 3
        self.number_of_obstacles = number_of_obstacles
        self.obstacles = obstacles_dict[obst_course_type]  # Specify the function for obstacles

    def safety_function(self, x):
        obstacle_info = self.obstacles()
        number_of_obstacles = obstacle_info.shape[0]
        ox = obstacle_info[:, 0]
        oy = obstacle_info[:, 1]
        r = obstacle_info[:, 2]
        
        if x.ndim == 1:
            numIter = 1
            h = np.zeros(number_of_obstacles)
            hx = np.zeros((number_of_obstacles, self.n))
            for ii in range(number_of_obstacles):
                h[ii] = (x[0]-ox[ii])**2 + (x[1]-oy[ii])**2 - r[ii]**2
                hx[ii] = np.concatenate((np.array([2*(x.item(0)-ox.item(ii)), 2*(x.item(1)-oy.item(ii))]), np.zeros((1, self.n - 2))), axis=None)  # for a single BaS!
                    
        else:
            numIter = x.shape[0]
            h = np.zeros((numIter, number_of_obstacles))
            hx = np.zeros((numIter, number_of_obstacles, self.n))
            for n_step in range(numIter):
                xk = x[n_step, :3]
                for ii in range(number_of_obstacles):
                    h[n_step, ii] = (xk[0]-ox[ii])**2 + (xk[1]-oy[ii])**2 - r[ii]**2
                    hx[n_step, ii] = np.concatenate((np.array([2*(xk.item(0)-ox.item(ii)), 2*(xk.item(1)-oy.item(ii))]), np.zeros((1, self.n - 2))), axis=None)  # for a single BaS!
                    # h[ii] = (x[0]-ox[ii])**4 + 1/10*(x[1]-oy[ii])**4 - r[ii]**2
                    # hx[ii] = np.concatenate((np.array([4*(x.item(0)-ox.item(ii))**3, 1/10*4*(x.item(1)-oy.item(ii))**3]), np.zeros((1, self.n - 2))), axis=None)  # for a single BaS!
        return h, hx


class Obstacles3D:
    def __init__(self, n, obst_course_type, number_of_obstacles=None):
        self.n = n
        self.number_of_obstacles = number_of_obstacles
        self.obstacles = obstacles3_dict[obst_course_type]  # Specify the function for obstacles

    # The input x is for the system quadrotor NOT including the barrier state
    def safety_function(self, x):
        obstacle_info = self.obstacles()
        number_of_obstacles = obstacle_info.shape[0]

        # Assume state position is are the last 3 indices
        sys_pos = x[-3:]
        obs_pos = obstacle_info[: ,0:3]
        r = obstacle_info[:, 3]
        h = np.zeros(number_of_obstacles)
        hx = np.zeros((number_of_obstacles, self.n))
        for ii in range(number_of_obstacles):
            h[ii] = np.linalg.norm(sys_pos.flatten()-obs_pos[ii,:])**2 - r[ii]**2
            hx[ii] = np.concatenate((np.zeros((1, self.n - 3)), 2*sys_pos.flatten()-obs_pos[ii,:] ), axis=None)
        return h, hx


def manual_obstacles3d_1():
    number_of_obstacles = 1
    obstacle_info = np.zeros((number_of_obstacles, 4))
    obstacle_info[0,:] = np.array([2, 2, 0, 1]) # x y z r
    return obstacle_info

def manual_obstacles3d_2():
    number_of_obstacles = 5
    obstacle_info = np.zeros((number_of_obstacles, 4))
    obstacle_info[0,:] = np.array([2,2,2,1]) # x y z r
    obstacle_info[1,:] = np.array([1,1,-1,0.5])
    obstacle_info[2,:] = np.array([0,0,0,1.3])
    obstacle_info[3,:] = np.array([-1,-1,-1,1])
    obstacle_info[4,:] = np.array([-2,-2,2,1.7])

    return obstacle_info

def manual_obstacles7():
    number_of_obstacles = 7
    obstacle_info = np.zeros((number_of_obstacles, 3))
    obstacle_info[0,:] = np.array([5,5,1]) # x y z r
    obstacle_info[1,:] = np.array([2,3,0.5])
    obstacle_info[2,:] = np.array([-1,-1,0.5])
    obstacle_info[3,:] = np.array([-2,1,1])
    obstacle_info[4,:] = np.array([5,7,0.5])
    obstacle_info[5, :] = np.array([4, 2, 1])
    obstacle_info[6, :] = np.array([0, 5, 1])

    return obstacle_info

def manual_obstacles1():
    number_of_obstacles = 3
    ox = np.zeros(number_of_obstacles)
    oy = np.zeros(number_of_obstacles)
    r = np.zeros(number_of_obstacles)
    ox[0], oy[0], r[0] = 0.5, 1.5, 0.3
    ox[1], oy[1], r[1] = 1.5, 0.2, 0.25
    ox[2], oy[2], r[2] = 2.5, 2.1, 0.32
    obstacle_info = np.zeros((number_of_obstacles, 3))
    for ii in range(number_of_obstacles):
        obstacle_info[ii, :] = np.concatenate([ox[ii], oy[ii], r[ii]], axis=None)
    return obstacle_info


def manual_obstacles2():
    number_of_obstacles = 10
    ox = np.zeros(number_of_obstacles)
    oy = np.zeros(number_of_obstacles)
    r = np.zeros(number_of_obstacles)
    ox[0], oy[0], r[0] = 0, -4, 1
    ox[1], oy[1], r[1] = 0, 4, 1
    ox[2], oy[2], r[2] = 0, -2, 1
    ox[3], oy[3], r[3] = 0, 6, 1
    ox[4], oy[4], r[4] = 0, 1.75, 1
    ox[5], oy[5], r[5] = 0, -8, 1
    ox[6], oy[6], r[6] = 0, -6, 1
    ox[7], oy[7], r[7] = 0, 8, 1
    ox[8], oy[8], r[8] = 0, -0.5, 1
    ox[9], oy[9], r[9] = 0, 5, 1
    obstacle_info = np.zeros((number_of_obstacles, 3))
    for ii in range(number_of_obstacles):
        obstacle_info[ii, :] = np.concatenate([ox[ii], oy[ii], r[ii]], axis=None)
    return obstacle_info

def manual_obstacles15():
    number_of_obstacles = 15
    ox = np.zeros(number_of_obstacles)
    oy = np.zeros(number_of_obstacles)
    r = np.zeros(number_of_obstacles)
    ox[0], oy[0], r[0] = -1, -0.25, 0.5
    ox[1], oy[1], r[1] = 0, 1, 0.5
    ox[2], oy[2], r[2] = -2, -1, 0.5
    ox[3], oy[3], r[3] = -3, -1, 0.5
    ox[4], oy[4], r[4] = -3, -2, 0.5
    ox[5], oy[5], r[5] = -2, 4, 0.5
    ox[6], oy[6], r[6] = -4, 1, 0.25
    ox[7], oy[7], r[7] = -4, 3, 1
    ox[8], oy[8], r[8] = -3.5, 2, 0.75
    ox[9], oy[9], r[9] = -1.5, 1, 0.25
    ox[10], oy[10], r[10] = 0, -2, 0.5
    ox[11], oy[11], r[11] = 0, 2, 0.5
    ox[12], oy[12], r[12] = -0.25, 3, 0.5
    ox[13], oy[13], r[13] = -1.25, 3, 1.5
    ox[14], oy[14], r[14] = -2.25, 1, 0.5

    obstacle_info = np.zeros((number_of_obstacles, 3))
    for ii in range(number_of_obstacles):
        obstacle_info[ii, :] = np.concatenate([ox[ii], oy[ii], r[ii]], axis=None)
    return obstacle_info


def manual_obstacles13():
    number_of_obstacles = 13
    ox = np.zeros(number_of_obstacles)
    oy = np.zeros(number_of_obstacles)
    r = np.zeros(number_of_obstacles)
    ox[0], oy[0], r[0] = 2.5, 2, 0.5
    ox[1], oy[1], r[1] = 4, 4, 0.5
    ox[2], oy[2], r[2] = 1, 2.5, 0.5
    ox[3], oy[3], r[3] = 4.5, 1, 0.5
    ox[4], oy[4], r[4] = 1, 4.5, 0.5
    ox[5], oy[5], r[5] = 2, 4.5, 0.5
    ox[6], oy[6], r[6] = 3, -1, 1.2
    ox[7], oy[7], r[7] = 2.0, 2.5, 0.5
    ox[8], oy[8], r[8] = 3, 3.5, 0.5
    ox[9], oy[9], r[9] = 1, -1, 0.5
    ox[10], oy[10], r[10] = 2.2, 0.5, 0.5
    ox[11], oy[11], r[11] = 3.0, 0.5, 0.5
    ox[12], oy[12], r[12] = 4, 2.5, 0.5
    obstacle_info = np.zeros((number_of_obstacles, 3))
    for ii in range(number_of_obstacles):
        obstacle_info[ii, :] = np.concatenate([ox[ii], oy[ii], r[ii]], axis=None)
    return obstacle_info


def random_obstacles(number_of_obstacles):
        # WIP
    # ox = np.zeros(number_of_obstacles)
    # oy = np.zeros(number_of_obstacles)
    # r = np.zeros(number_of_obstacles)
    obstacle_info = np.zeros((number_of_obstacles, 3))
    for ii in range(number_of_obstacles):
        obstacle_info[ii, :] = np.concatenate([random(), random(), random()], axis=None)
    return obstacle_info


obstacles_dict = {'manual_obstacles1': manual_obstacles1,
                  'manual_obstacles2': manual_obstacles2,
                  'manual_obstacles7' : manual_obstacles7,
                  'manual_obstacles13': manual_obstacles13,
                  'manual_obstacles15': manual_obstacles15}

obstacles3_dict = {'manual_obstacles1': manual_obstacles3d_1,
                   'manual_obstacles2': manual_obstacles3d_2}