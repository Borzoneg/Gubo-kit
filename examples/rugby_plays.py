import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class RugbyPlaysPlotter:
    def __init__(self, figsize: tuple[float, float] = (16, 9), title="Rugby plays", radius_player: float = 1):
        plt.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor('green')
        self.fig.canvas.manager.set_window_title(title)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.axis('off')
        
        self.colors = {'team': 'dodgerblue', 'opp': 'crimson'}
        self.radius_player = radius_player
        self.min_x, self.max_x, self.min_y, self.max_y = float('inf'), -float('inf'), float('inf'), -float('inf')        

        cx = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
        cy = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
        print(cx, cy)
        

        self.ax.set_xlim(0, 30)  # Set x and y limits
        self.ax.set_ylim(-20, 20)       

        self.ax.set_aspect('equal', adjustable='box')
        self.players = {}
        self.ball = []
    
    def add_play(self, filename="./files/Rugby_plays/play.json"):
        with open(filename, 'r') as f:
            play_dict = json.load(f)
            self.populate_players_queue(play_dict['players'], play_dict['t'])
            self.populate_ball_queue(play_dict['ball'], play_dict['t'])

    def populate_ball_queue(self, balldict, end_t):
        for t in range(end_t):
            self.ball.append((0,0))

    def populate_players_queue(self, players, end_t):
        self.players = {}
        for t in range(end_t):
            for pkey in players:
                p = players[pkey]
                if t == 0:
                    player_patch = self.create_player((0,0), p['team'])
                    self.players[pkey] = {"pos":[], "patch": player_patch}
                p_start = np.array(p['start'])
                p_movement = self.find_p_xy_at_t(t, p['speeds'])
                current_pos_p = (p_start + p_movement)
                self.players[pkey]["pos"].append(current_pos_p)

    def find_p_xy_at_t(self, t, speeds):
        direction_dict = {'n': np.array((0, -1)), 'ne': np.array((1, -1)), 'e': np.array((1, 0)), 'se': np.array((1, 1)),
                          's': np.array((0, 1)), 'sw': np.array((-1, 1)), 'w': np.array((-1, 0)), 'nw': np.array((-1, -1))}
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

    def create_player(self, xy, team):
        player = patches.Circle(np.array(xy), radius=self.radius_player, color=self.colors[team])
        self.ax.add_patch(player)
        return player

    def update(self):
        if len(self.ball) > 0:
            for pkey in self.players:
                new_pos = self.players[pkey]["pos"].pop()
                self.players[pkey]['patch'].set_center(new_pos)  # Update position
            self.ball.pop()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
            

if __name__ == "__main__":
    rugby_plays_plotter = RugbyPlaysPlotter()
    input("Enter to start")
    rugby_plays_plotter.add_play()
    while True:
        rugby_plays_plotter.update()
