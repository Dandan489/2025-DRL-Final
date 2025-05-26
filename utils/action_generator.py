from collections import namedtuple
import numpy as np
'''
Source Unit	[0,hxw-1]: the location of the unit selected to perform an action
Action Type	[0,5]: NOOP, move, harvest, return, produce, attack
Move Parameter	[0,3]: north, east, south, west
Harvest Parameter [0,3]: north, east, south, west
Return Parameter [0,3]: north, east, south, west
Produce Direction Parameter	[0,3]: north, east, south, west
Produce Type Parameter	[0,6]: resource, base, barrack, worker, light, heavy, ranged
Relative Attack Position [0,ar^2-1]: the relative location of the unit that will be attacked
'''
class Action_Generator:
    def __init__(self, h, w):
        # coordinate start from left-top corner
        self.h = h
        self.w = w
        self.action_plane = np.full((h, w, 7), [0,0,0,0,0,0,0])
    def flatten_action(self): #we need to submit a 1-d array to the environment position : [(0,0), (0,1), (0,2)...,(1, 0)..., (w-1, h-1)]. Weird, order by y
        ret = np.full((self.h*self.w, 7), [0,0,0,0,0,0,0])
        for x in range(self.w):
            for y in range(self.h):
                ret[x * self.w + y] = self.action_plane[y, x]
        return ret

    def reset_action(self):
        self.action_plane = np.full((self.h, self.w, 7), [0,0,0,0,0,0,0])
    
    def perform_on_grid(self, y, x, action):
        #assert action is a list/array of length 7
        self.action_plane[y][x] = action