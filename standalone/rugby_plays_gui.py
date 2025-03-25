import tkinter as tk
import numpy as np
import json
import sys
import os

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
        print(f"player[{str(self.name):2s} : [{int(new_xy[0])}, {int(new_xy[1])}]]")
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r, -self.r, self.r, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)
        self.canvas.coords(self.text, *new_xy)

    def flush_queue(self):
        self.xy = []
    
class Arrow:
    def __init__(self, canvas, label):
        self.canvas = canvas
        self.xy = []
        self.label = label
        self.arrow = None

    def draw_arrow(self, start_xy, end_xy):
        self.arrow = self.canvas.create_line(start_xy[0], start_xy[1], end_xy[0], end_xy[1], arrow=tk.LAST, width=2, tags=self.label)
        self.text = self.canvas.create_text(start_xy[0], start_xy[1], text=self.label, font=("Arial", 9))
        self.txt_bg = self.canvas.create_rectangle(self.canvas.bbox(self.text), fill="white", outline="white")
        self.canvas.tag_raise(self.text)

    def add_to_queue(self, start_pos, end_pos):
        self.xy.append([start_pos, end_pos])
    
    def extend_queue(self, xy):
        self.xy.extend(xy)

    def next_step(self):
        new_xy = self.xy.pop(0)
        print(f"arrow {new_xy}")
        if new_xy == [-1, -1]:
            self.canvas.itemconfig(self.arrow, state=tk.HIDDEN)
            self.canvas.itemconfig(self.text, state=tk.HIDDEN)
            self.canvas.itemconfig(self.txt_bg, state=tk.HIDDEN)
        else:
            self.canvas.itemconfig(self.txt_bg, state=tk.NORMAL)
            self.canvas.itemconfig(self.text, state=tk.NORMAL)
            self.canvas.itemconfig(self.arrow, state=tk.NORMAL)
            self.canvas.coords(self.arrow, new_xy[0][0], new_xy[0][1], new_xy[1][0], new_xy[1][1])
            x_pad = ((self.canvas.bbox(self.text)[2] - self.canvas.bbox(self.text)[0])//2) * np.sign(new_xy[0][0] - new_xy[1][0])
            y_pad = ((self.canvas.bbox(self.text)[3] - self.canvas.bbox(self.text)[1])//2) * np.sign(new_xy[0][1] - new_xy[1][1])
            self.canvas.coords(self.text, new_xy[0][0]+x_pad, new_xy[0][1]+y_pad)
            self.canvas.coords(self.txt_bg, self.canvas.bbox(self.text))

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
        print(f"ball: [{new_xy[0]:08.3f}, {new_xy[1]:08.3f}]")
        new_circle_coord = (np.hstack((new_xy, new_xy)) + np.array([-self.r*1.2, -self.r, self.r*1.2, self.r]))
        self.canvas.coords(self.circle, *new_circle_coord)

    def flush_queue(self):
        self.xy = []

class RugbyPlaysPlotter:
    def __init__(self, root, size=(1500, 2000), title="Rugby plays", radius_player=10, start_point=(0, 0.5)):
        self.root = root
        self.size = size
        self.scaling = 1
        self.pitch_w = 700 * self.scaling
        self.pitch_h = 1200 * self.scaling
        self.start_point = np.array(start_point) * np.array([self.pitch_w, self.pitch_h])
        self.radius_player = radius_player

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side="top", pady=(50, 0))
        self.canvas = tk.Canvas(self.canvas_frame, width=self.pitch_w, height=self.pitch_h, bg='green')
        self.canvas.pack(side="top")
        
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(side="top")
        self.add_play_btn = tk.Button(self.btn_frame, text="add play", command=self.add_play)
        self.add_play_btn.pack(side='left', padx=10)
        self.pause_play_btn = tk.Button(self.btn_frame, text="start", command=self.pause_start_play, width=5)
        self.pause_play_btn.pack(side='left', padx=10)
        self.step_play_btn = tk.Button(self.btn_frame, text="step", command=self.step_play)
        self.step_play_btn.pack(side='left', padx=10)
        self.filepath_entry = tk.Entry(self.btn_frame, textvariable=tk.StringVar(value="./files/Rugby_plays/rocknroll12.json"), width=30)
        self.filepath_entry.pack(side='left', padx=10)
        self.t_label = tk.Label(self.btn_frame, text="----")
        self.t_label.pack(side='left', padx=10)
        
        self.draw_pitch()
        
        self.players = {}
        self.arrows = []
        self.ball = Ball(self.canvas, self.radius_player//2)
        self.t = 0
        self.paused = True
        self.step = False

    def draw_pitch(self):
        self.canvas.create_line(0, self.pitch_h // 2, self.pitch_w, self.pitch_h // 2, fill='white', width=2)  # Midline

    def add_player(self, name, start, team):
        self.players[name] = Player(self.canvas, name, start, self.radius_player, team)

    def add_play(self, filename=None):
        if filename is None:
            filename = self.filepath_entry.get()
        self.canvas.delete("all")
        self.canvas.create_line(0, self.pitch_h // 2, self.pitch_w, self.pitch_h // 2, fill='white', width=2)  # Midline
        self.t = 0
        with open(filename, 'r') as f:
            play_dict = json.load(f)
            speed = play_dict['speed']
            self.populate_players_queue(play_dict['players'], play_dict['t'], step_t=speed)
            self.populate_ball_queue(play_dict['ball'], step_t=speed)
            try:
                self.populate_arrow_queue(play_dict['arrows'], end_t=play_dict['t'], step_t=speed)
            except:
                self.arrows = []
                pass

    def pause_start_play(self):
        self.paused = not self.paused
        new_label = "pause" if not self.paused else "play"
        self.pause_play_btn.config(text=new_label)
    
    def step_play(self):
        self.step = True

    def populate_ball_queue(self, balldict, step_t=1):
        self.ball.flush_queue()
        possessions = balldict['possession']
        last_step = 0
        for player_posession, idxs in possessions:
            start_poss, end_poss = idxs[0]//step_t, idxs[1]//step_t
            poss_p_xy = [p_xy for p_xy in self.players[player_posession].xy[start_poss : end_poss]]
            if start_poss == 0:
                self.ball.draw_ball(poss_p_xy[0])
            if start_poss != last_step:
                # ball_in_air = [[100, 500] for _ in range(int(start_poss - last_step))]
                ball_in_air = self.find_move_b_at_t(ball_xy=self.ball.xy[-1], target_xy=self.players[player_posession].xy[start_poss], steps=int(start_poss - last_step))
                self.ball.xy.extend(ball_in_air)
            ball_in_possession = [p_xy for p_xy in self.players[player_posession].xy[start_poss : end_poss+1]]
            self.ball.xy.extend(ball_in_possession)
            last_step = end_poss+1
        
    def populate_players_queue(self, players, end_t, step_t=1):
        self.players = {}
        for t in np.arange(0, end_t, step_t):
            for pkey in players:
                p = players[pkey]
                p_start = (np.array(p['start']) + np.full(2, self.radius_player) + self.start_point)
                p_movement = self.find_p_xy_at_t(t, p['speeds'])
                current_pos_p = (p_start + p_movement) 
                if t == 0:
                    self.players[pkey] = Player(canvas=self.canvas, name=pkey, r=self.radius_player, start=p_start, team=p['team'])
                self.players[pkey].add_to_queue(current_pos_p)
                # self.players[pkey]["pos"].append(current_pos_p)
    
    def populate_arrow_queue(self, arrows_list, end_t, step_t=1):
        self.arrows = []
        for arrow_dict in arrows_list:
            arrow = Arrow(self.canvas, label=arrow_dict['label'])
            arrow.draw_arrow([0, 0], [0, 0])
            # self.arrows.append(arrow)
            arrow.extend_queue([[-1, -1]] * arrow_dict['spawn_t'])
            for start_xys, end_xys in zip(arrow_dict['start_arrow'], arrow_dict['end_arrow']):
                arrow.extend_queue([[start_xys[0], end_xys[0]]] * start_xys[1])
            arrow.extend_queue([[-1, -1]] * (end_t - (arrow_dict['spawn_t'] + arrow_dict['duration_t'])))
            self.arrows.append(arrow)
            
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
        if not self.paused or self.step:
            self.step = False
            if len(self.ball.xy) > 0:
                print(f"{self.t:03d}:", end=" ")
                self.ball.next_step()
                for pkey in self.players:
                    self.players[pkey].next_step()  # Update position
                self.t_label.config(text=f"{self.t:04d}")
                for arrow in self.arrows:
                    arrow.next_step()
                self.t += 1
                print("="*100)
            
    def run(self):
        self.update()
        self.root.after(100, self.run)  # Update every 100ms

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Rugby Play Simulation")
    rugby_plotter = RugbyPlaysPlotter(root)
    if len(sys.argv) > 1:
        rugby_plotter.add_play(os.path.join("./files/Rugby_plays", f"{sys.argv[1]}.json"))
        rugby_plotter.pause_play_btn.config(text="pause")
        rugby_plotter.paused = False

    # rugby_plotter.add_player('Player1', (100, 100), 'team')
    # rugby_plotter.add_player('Player2', (200, 200), 'opp')

    rugby_plotter.run()
    root.mainloop()
