from collections import namedtuple
import numpy as np
'''
Hit Points	5	0, 1, 2, 3, ≥ 4
Resources	5	0, 1, 2, 3, ≥ 4
Owner	3	-,player 1, player 2
Unit Types	8	-, resource, base, barrack, worker, light, heavy, ranged
Current Action	6	-, move, harvest, return, produce, attack
Terrain	2	free, wall
'''
grid_data = namedtuple("grid_data", "hit_points, resources, owner, unit_types, current_action, terrain")

class Obs_Parser:
    unit_type_dict = {0: "None", 1: "resource", 2: "base", 3: "barrack", 4: "worker", 5: "light", 6: "heavy", 7: "ranged"}
    action_dict = {0: "None", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}

    def __init__(self, vec_obs):
        self.vec_obs = vec_obs #shape: (num_selfplay_env, h, w, 29)
        self.num_env, self.h, self.w, self.c = vec_obs.shape
        self.obs = [] #for each env, parse h*w grid_data
    def parse_feature(self, feature):
        return grid_data(np.argmax(feature[0:5]), np.argmax(feature[5:10]), np.argmax(feature[10:13]), np.argmax(feature[13:21]), np.argmax(feature[21:27]), np.argmax(feature[27:29]))

    def parse(self, vec_obs): #parse 
        self.vec_obs = vec_obs
        self.obs = []
        for e in range(self.num_env):
            plane_data = [[] for _ in range(self.h)]
            for i in range(self.h):
                for j in range(self.w):
                    plane_data[i].append(self.parse_feature(vec_obs[e][i][j]))
            self.obs.append(plane_data)
        return self.obs