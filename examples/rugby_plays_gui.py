import tkinter as tk
import numpy as np
import json

class Player:
    def __init__(self, canvas, name, start, r, team):
        self.canvas = canvas
        self.name = name
        self.xy = []
        self.r = r
        self.color = {'team': 'dodgerblue', 'opp': 'crimson'}[team]
        start_coord = (np.hstack((start, start)) + np.array([-self.r, -self.r, self.r, self.r]))
        self.circle = self.canvas.create_oval(*start_coord, fill=self.color, outline='black')
        self.text = self.canvas.create_text(*start, text=self.name, fill='white', font=('Helvetica', 10, 'bold'))
        
    def add_to_queue(self, new_pos):
        self.xy.insert(0, new_pos)

    def next_step(self):
        new_xy = self.xy.pop()
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r, -self.r, self.r, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)
        self.canvas.coords(self.text, *new_xy)
    
class Ball:
    def __init__(self, canvas, r):
        self.canvas = canvas
        self.xy = []
        self.r = r       

    def draw_ball(self, xy):
        xy_coord = (np.hstack((xy, xy)) + np.array([-self.r*1.2, -self.r, self.r*1.2, self.r]))
        self.circle = self.canvas.create_oval(*xy_coord, fill='brown', outline='black')
    
    def add_to_queue(self, new_pos):
        self.xy.insert(0, new_pos)

    def next_step(self):
        new_xy = self.xy.pop()
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r*1.2, -self.r, self.r*1.2, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)

class RugbyPlaysPlotter:
    # TODO: pause button
    def __init__(self, root, size=(700, 1000), title="Rugby plays", radius_player=10, start_point=(0, 0.5)):
        self.root = root
        self.size = size
        self.start_point = np.array(start_point) * np.array(self.size)

        self.canvas = tk.Canvas(self.root, width=self.size[0], height=self.size[1], bg='green')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.radius_player = radius_player
        # self.btn_frame = tk.Button(self.root, text="start play", command=self.start_play)
        self.btn = tk.Button(self.root, text="start play", command=self.start_play)
        self.btn.pack(side='left', padx=10)
        self.filepath_entry = tk.Entry(self.root, textvariable=tk.StringVar(value="./files/Rugby_plays/play.json"), width=30)
        self.filepath_entry.pack(side='left', padx=10)
        
        self.canvas.create_line(0, self.size[1] // 2, self.size[0], self.size[1] // 2, fill='white', width=2)  # Midline
        self.players = {}
        self.ball = Ball(self.canvas, self.radius_player//2)

    def add_player(self, name, start, team):
        self.players[name] = Player(self.canvas, name, start, self.radius_player, team)

    def start_play(self):
        filename = self.filepath_entry.get()
        self.canvas.delete("all")
        self.canvas.create_line(0, self.size[1] // 2, self.size[0], self.size[1] // 2, fill='white', width=2)  # Midline
        with open(filename, 'r') as f:
            play_dict = json.load(f)
            speed = play_dict['speed']
            self.populate_players_queue(play_dict['players'], play_dict['t'], step_t=speed)
            self.populate_ball_queue(play_dict['ball'], play_dict['t'], step_t=speed)

    def populate_ball_queue(self, balldict, end_t, step_t=1):
        for t in np.arange(0, end_t, step_t):
            # need to think about how to do this ???
            if t == 0:
                self.ball.draw_ball(balldict["start"])
                self.ball.add_to_queue(balldict["start"]+ self.start_point)
                continue
            self.ball.add_to_queue(balldict["start"]+ self.start_point)
            for key in balldict:
                pass

    def populate_players_queue(self, players, end_t, step_t=1):
        self.players = {}
        for t in np.arange(0, end_t, step_t):
            for pkey in players:
                p = players[pkey]
                p_start = np.array(p['start']) + np.full(2, self.radius_player) + self.start_point
                p_movement = self.find_p_xy_at_t(t, p['speeds'])
                current_pos_p = (p_start + p_movement)
                if t == 0:
                    self.players[pkey] = Player(canvas=self.canvas, name=pkey, r=self.radius_player, start=p_start, team=p['team'])

                self.players[pkey].add_to_queue(current_pos_p)
                # self.players[pkey]["pos"].append(current_pos_p)

    def find_p_xy_at_t(self, t, speeds):
        direction_dict = {'n': np.array((0, -1)), 'ne': np.array((1, -1)), 'e': np.array((1, 0)), 'se': np.array((1, 1)),
                          's': np.array((0, 1)), 'sw': np.array((-1, 1)), 'w': np.array((-1, 0)), 'nw': np.array((-1, -1)), 'o': np.array((0, 0))}
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
        return (self.size[0] // 2, self.size[1] // 2)

    def update(self):
        if len(self.ball.xy) > 0:
            self.ball.next_step()
            for pkey in self.players:
                # print("Player step")
                self.players[pkey].next_step()  # Update position
            # print("ball step")
            
    def run(self):
        self.update()
        self.root.after(100, self.run)  # Update every 100ms

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Rugby Play Simulation")
    rugby_plotter = RugbyPlaysPlotter(root)

    # rugby_plotter.add_player('Player1', (100, 100), 'team')
    # rugby_plotter.add_player('Player2', (200, 200), 'opp')

    rugby_plotter.run()
    root.mainloop()
