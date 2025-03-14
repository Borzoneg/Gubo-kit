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
        self.xy.append(new_pos)

    def next_step(self):
        new_xy = self.xy.pop(0)
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r, -self.r, self.r, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)
        self.canvas.coords(self.text, *new_xy)

    def flush_queue(self):
        self.xy = []
    
class Ball:
    def __init__(self, canvas, r):
        self.canvas = canvas
        self.xy = []
        self.r = r  

    def draw_ball(self, xy):
        xy_coord = (np.hstack((xy, xy)) + np.array([-self.r*1.2, -self.r, self.r*1.2, self.r]))
        self.circle = self.canvas.create_oval(*xy_coord, fill='brown', outline='black')
    
    def add_to_queue(self, new_pos):
        self.xy.append(new_pos)

    def next_step(self):
        new_xy = self.xy.pop(0)
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r*1.2, -self.r, self.r*1.2, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)

    def flush_queue(self):
        self.xy = []

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
        self.add_play_btn = tk.Button(self.root, text="add play", command=self.add_play)
        self.add_play_btn.pack(side='left', padx=10)
        self.pause_play_btn = tk.Button(self.root, text="start", command=self.pause_start_play, width=5)
        self.pause_play_btn.pack(side='left', padx=10)
        self.step_play_btn = tk.Button(self.root, text="step", command=self.step_play)
        self.step_play_btn.pack(side='left', padx=10)
        self.filepath_entry = tk.Entry(self.root, textvariable=tk.StringVar(value="./files/Rugby_plays/rocknroll12.json"), width=30)
        self.filepath_entry.pack(side='left', padx=10)
        self.t_label = tk.Label(self.root, text="----")
        self.t_label.pack(side='left', padx=10)
        
        self.canvas.create_line(0, self.size[1] // 2, self.size[0], self.size[1] // 2, fill='white', width=2)  # Midline
        self.players = {}
        self.ball = Ball(self.canvas, self.radius_player//2)
        self.t = 0
        self.paused = True
        self.step = False

    def add_player(self, name, start, team):
        self.players[name] = Player(self.canvas, name, start, self.radius_player, team)

    def add_play(self):
        filename = self.filepath_entry.get()
        self.canvas.delete("all")
        self.canvas.create_line(0, self.size[1] // 2, self.size[0], self.size[1] // 2, fill='white', width=2)  # Midline
        self.t = 0
        with open(filename, 'r') as f:
            play_dict = json.load(f)
            speed = play_dict['speed']
            self.populate_players_queue(play_dict['players'], play_dict['t'], step_t=speed)
            self.populate_ball_queue(play_dict['ball'], play_dict['t'], step_t=speed)

    def pause_start_play(self):
        self.paused = not self.paused
        new_label = "pause" if not self.paused else "play"
        self.pause_play_btn.config(text=new_label)
    
    def step_play(self):
        self.step = True

    def populate_ball_queue(self, balldict, end_t, step_t=1):
        self.ball.flush_queue()
        possessions = balldict['possession']
        last_step = 0
        for player_posession in possessions:
                poss_p_xy = [p_xy for p_xy in self.players[player_posession].xy[possessions[player_posession][0] : possessions[player_posession][1]]]
                if possessions[player_posession][0] == 0:
                    self.ball.draw_ball(poss_p_xy[0])
                if possessions[player_posession][0] != last_step:
                    # ball_in_air = [[100, 500] for _ in range(int(possessions[player_posession][0] - last_step))]
                    ball_in_air = self.find_move_b_at_t(ball_xy=self.ball.xy[-1], target_xy=self.players[player_posession].xy[possessions[player_posession][0]], steps=int(possessions[player_posession][0] - last_step))
                    self.ball.xy.extend(ball_in_air)
                ball_in_possession = [p_xy for p_xy in self.players[player_posession].xy[possessions[player_posession][0] : possessions[player_posession][1]+1]]
                self.ball.xy.extend(ball_in_possession)
                last_step = possessions[player_posession][1]+1
        
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
        direction_dict = {  'e'  : -0 * np.pi / 1,
                            'nee': -1 * np.pi / 6,
                            'ne' : -1 * np.pi / 4,
                            'nne': -1 * np.pi / 3,
                            'n'  : -1 * np.pi / 2,
                            'nnw': -2 * np.pi / 3,
                            'nw' : -3 * np.pi / 4,
                            'nww': -5 * np.pi / 6,
                            'w'  : -1 * np.pi / 1,
                            'sww': -7 * np.pi / 6,
                            'sw' : -5 * np.pi / 4,
                            'ssw': -4 * np.pi / 3,
                            's'  : -3 * np.pi / 2,
                            'sse': -5 * np.pi / 3,
                            'se' : -7 * np.pi / 4,
                            'see': -11 * np.pi / 6}

        movement = np.zeros(2)
        for s in speeds:
            direction = np.array((np.cos(direction_dict[s[0]]), np.sin(direction_dict[s[0]]))) if s[0] != 'o' else np.array((0, 0))
            if t >= s[1]: # if t is above the end time of this speed we use all the speed and remove it from the t 
                movement += direction * s[1]
                t -= s[1]
            else:
                movement += direction * (t)
                break
        return movement

    def find_move_b_at_t(self, ball_xy, target_xy, steps):
        return np.linspace(ball_xy, target_xy, steps)

    def update(self):
        if not self.paused:
            if len(self.ball.xy) > 0:
                self.ball.next_step()
                for pkey in self.players:
                    self.players[pkey].next_step()  # Update position
                self.t_label.config(text=f"{self.t:04d}")
                self.t += 1
        if self.step:
            if len(self.ball.xy) > 0:
                self.ball.next_step()
                for pkey in self.players:
                    self.players[pkey].next_step()  # Update position
                self.t_label.config(text=f"{self.t:04d}")
                self.t += 1
            self.step = False
            
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
