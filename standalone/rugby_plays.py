import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class PlayerPatch:
    def __init__(self, ax, name, start, r, team):
        self.ax = ax
        self.xy = []
        self.r = r
        self.color = {'team': 'dodgerblue', 'opp': 'crimson'}[team]
        self.patch = patches.Circle(np.array(start), radius=self.r, facecolor=self.color, edgecolor='black', linewidth=2)
        self.ax.add_patch(self.patch)
        self.text = self.ax.text(start[0], start[1], name, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    def add_to_queue(self, new_pos):
        self.xy.insert(0, new_pos)

    def next_step(self):
        new_xy = self.xy.pop()
        self.text.set_position(new_xy)
        self.patch.set_center(new_xy)
        return 

class RugbyPlaysPlotter:
    def __init__(self, figsize: tuple[float, float] = (6, 8), title="Rugby plays", radius_player: float = 1):
        # plt.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_facecolor('green')
        self.fig.canvas.manager.set_window_title(title)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.axis('off')
        # the background patch is part of the axis so we need it back
        self.ax.add_artist(self.ax.patch) 
        self.ax.patch.set_zorder(-1)
        
        self.colors = {'team': 'dodgerblue', 'opp': 'crimson'}
        self.radius_player = radius_player
        self.min_x, self.max_x, self.min_y, self.max_y = float('inf'), -float('inf'), float('inf'), -float('inf')        

        
        self.ax.set_xlim(0, 30)  # Set x and y limits
        self.ax.set_ylim(-20, 20)       

        cx = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
        cy = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
        # print(cx, cy)
        print(self.ax.get_xlim()[0], self.ax.get_xlim()[1], self.ax.get_ylim()[0], self.ax.get_ylim()[1])

        self.ax.set_aspect('equal', adjustable='box')
        self.players = {}
        self.ball = []
    
    def add_play(self, filename="./files/Rugby_plays/play.json"):
        with open(filename, 'r') as f:
            play_dict = json.load(f)
            speed = 0.1 # add it to jsone
            self.populate_players_queue(play_dict['players'], play_dict['t'], step_t=speed)
            self.populate_ball_queue(play_dict['ball'], play_dict['t'], step_t=speed)

    def populate_ball_queue(self, balldict, end_t, step_t=1):
        for t in np.arange(0, end_t, step_t):
            self.ball.append((0,0))

    def populate_players_queue(self, players, end_t, step_t=1):
        self.players = {}
        for t in np.arange(0, end_t, step_t):
            for pkey in players:
                p = players[pkey]
                    # player_patch = self.create_player((0,0), p['team'], pkey)
                    # self.players[pkey] = {"pos":[], "patch": player_patch}
                p_start = np.array(p['start']) + np.full(2, self.radius_player)
                p_movement = self.find_p_xy_at_t(t, p['speeds'])
                current_pos_p = (p_start + p_movement)
                if t == 0:
                    print(p_start)
                    print(p_start+np.full(2, self.radius_player))
                    self.players[pkey] = PlayerPatch(ax=self.ax, name=pkey, r=self.radius_player, start=p_start+np.full(2, self.radius_player), team=p['team'])
                self.players[pkey].add_to_queue(current_pos_p)
                # self.players[pkey]["pos"].append(current_pos_p)

    def find_p_xy_at_t(self, t, speeds):
        direction_dict = {'n': np.array((0, 1)), 'ne': np.array((1, 1)), 'e': np.array((1, 0)), 'se': np.array((1, -1)),
                          's': np.array((0, -1)), 'sw': np.array((-1, -1)), 'w': np.array((-1, 0)), 'nw': np.array((-1, 1)), 'o': np.array((0, 0))}
        movement = np.zeros(2)
        for s in speeds:
            if t >= s[1]: # if t is above the end time of this speed we use all the speed and remove it from the t 
                movement += direction_dict[s[0]] * s[1]
                t -= s[1]
            else:
                movement += direction_dict[s[0]] * (t)
                break
        return movement

    def find_b_xy_at_t(self, target):
        print(target)

    

    def update(self):
        if len(self.ball) > 0:
            for pkey in self.players:
                self.players[pkey].next_step()  # Update position
            self.ball.pop()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            

if __name__ == "__main__":
    rugby_plays_plotter = RugbyPlaysPlotter()
    # input("Enter to start")
    rugby_plays_plotter.add_play()
    while True:
        rugby_plays_plotter.update()
