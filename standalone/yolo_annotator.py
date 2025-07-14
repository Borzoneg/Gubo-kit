import tkinter as tk
import spatialmath as sm
import numpy as np
from tkinter import ttk
import cv2
import PIL
import numpy as np
from ultralytics import YOLO
import PIL.ImageTk
import PIL.Image
import os

class _BoundingBoxEditor:
    def __init__(self, canvas, frame, tag='', move_color='blue'):
        self.canvas = canvas
        self.selected_box = None
        self.dragging = None  # "move" or "resize"
        self.start_x = 0
        self.start_y = 0
        self.frame = frame
        self.label = None
        self.tag = tag
        self.move_color = move_color

        self.boxes_items = []  # Store drawn objects
        self.bbs_position = []

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.editable = True

    def set_color_icon(self, color: str):
        self.move_color = color

    def set_label(self, label):
        self.label = label

    def add_bb(self, bb):
        self.bbs_position.append(bb)

    def add_bbs(self, bbs):
        for bb in bbs:
            self.add_bb(bb)
    
    def clear_bbs(self):
        self.bbs_position = []
        self.canvas.delete('bbs' + self.tag)
        self.boxes_items.clear()

    def lock(self):
        self.editable = False
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def _draw_box(self, label, x_min=500, y_min=500, x_max=1000, y_max=1000, scale=1, padx=0, pady=0):
        x_min = ((x_min * scale) + padx)
        x_max = ((x_max * scale) + padx)
        y_min = ((y_min * scale) + pady)
        y_max = ((y_max * scale) + pady)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        self.canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="black", width=2, tags='bbs' + self.tag)
        self.canvas.create_rectangle(x_max-len(label)*10, y_max-9, x_max, y_max+9, fill="white", outline="white", tags='bbs' + self.tag) # label bg
        self.canvas.create_text(x_max-len(label)*5, y_max, text=label, font=("Arial", 10), tags='bbs' + self.tag) # label txt
        if self.editable:
            self.canvas.create_rectangle(x_max-7, y_min-7, x_max+7, y_min+7, fill="white", outline="red", tags='bbs' + self.tag) # delete_bg
            move_handle = self.canvas.create_oval(center_x-5, center_y-5, center_x+5, center_y+5, fill=self.move_color, tags='bbs' + self.tag)
            resize_handle = self.canvas.create_rectangle(x_min-5, y_min-5, x_min+5, y_min+5, fill="red", tags='bbs' + self.tag)
            delete_handle = self.canvas.create_text(x_max, y_min, text="X", fill="red", font=("Arial", 10), tags = 'bbs' + self.tag)
            self.boxes_items.append([move_handle, resize_handle, delete_handle])
        self.canvas.lift("bbs" + self.tag)

    def draw_boxes(self, scale=1, padx=0, pady=0):
        """Draws bounding boxes with resize/move handles"""
        self.canvas.delete('bbs' + self.tag)
        self.boxes_items.clear()
        for i, bb in enumerate(self.bbs_position):
            label = self.label if self.label is not None else f"{i:02d}"
            self._draw_box(label, *bb, scale=scale, padx=padx, pady=pady)
        
        if hasattr(self.frame, "number_label"):
            self.frame.number_label.configure(text=f"\nNumber of cells: {len(self.bbs_position)}\n")

    def on_click(self, event):
        """Detects which part of a box was clicked (move/resize)"""
        if self.canvas.find_withtag(tk.CURRENT):  # If clicked on an item
            item = self.canvas.find_withtag(tk.CURRENT)[0]
            for i, [move_handle, resize_handle, delete_handle] in enumerate(self.boxes_items):
                if item == move_handle:
                    self.selected_box = i
                    self.dragging = "move"
                    self.start_x, self.start_y = event.x, event.y
                    break   
                elif item == resize_handle:  # Resize box
                    self.selected_box = i
                    self.dragging = "resize"
                    self.start_x, self.start_y = event.x, event.y
                    break  
                elif item == delete_handle: # if clicked on close button, remove bb
                    if len(self.bbs_position) > 1:
                        self.bbs_position.pop(i)
                        self.draw_boxes()
                        break
                       
    def on_drag(self, event):
        """Moves or resizes the selected bounding box"""
        if self.selected_box is None:
            return
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        x_min, y_min, x_max, y_max = self.bbs_position[self.selected_box]

        if self.dragging == "move":
            x_min += dx
            y_min += dy
            x_max += dx
            y_max += dy

        elif self.dragging == "resize":
            x_min += dx
            y_min += dy

        self.bbs_position[self.selected_box] = (x_min, y_min, x_max, y_max)

        self.start_x = event.x
        self.start_y = event.y
        self.draw_boxes()

    def on_release(self, event):
        """Resets after dragging"""
        self.selected_box = None
        self.dragging = None

class MemGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.idx = 0
        self.foldername = ''
        self.layout_gui()
        self.bb_drawer = _BoundingBoxEditor(self.home_frame.canvas, self.home_frame)
        
        self.import_model()
        self.save_foldername()

        self.img = cv2.imread(os.path.join(self.foldername, f'pic{self.idx:02d}.png'))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        self.img = PIL.Image.fromarray(self.img)

    def layout_gui(self):
        self.title("MeM use case")
        self.geometry("1280x720")

        # self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.configure(bg="#4b5661")

        # self.top_frame = tk.Frame(self, bg='#1e2a38')
        # self.top_frame.grid(row=0, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        # self.top_frame.grid_columnconfigure(0, weight=1)
        # self.top_frame.grid_columnconfigure(1, weight=2)
        # self.top_frame.grid_columnconfigure(1, weight=2)
        self.mid_frame = tk.Frame(self, bg="#768799")
        self.mid_frame.grid(row=0, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        self.mid_frame.grid_columnconfigure(0, weight=1)
        self.mid_frame.grid_rowconfigure(0, weight=1)
        self.bot_frame = tk.Frame(self, bg="#1e2a38")
        self.bot_frame.grid(row=1, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        [self.bot_frame.grid_rowconfigure(i, weight=1, minsize=10) for i in range(3)]
        self.bot_frame.grid_columnconfigure(0, weight=1)
        self.bot_frame.grid_columnconfigure(1, weight=1)
        # self.bot_frame.grid_columnconfigure(2, weight=1)
                
        self.home_frame = HomeScreen(self.mid_frame, self)
        self.home_frame.config(background='#2e3f4f')
        self.home_frame.grid(row=0, column=0, sticky='nsew', padx=(5, 5), pady=(5, 5))
        self.home_frame.rowconfigure(0, weight=1)
        self.home_frame.columnconfigure(0, weight=1)
        self.home_frame.grid_propagate(False)

        self.model_label = tk.Label(self.bot_frame, text="Model:")
        self.model_label.grid(row=0, column=0, sticky='nsew')
        self.model_entry = tk.Entry(self.bot_frame)
        self.model_entry.insert(0, '/home/gu/fluently_ws/fluently_mem/data/cells_best_model.pt')
        self.model_entry.grid(row=0, column=1, sticky='nsew')
        self.model_btn = tk.Button(self.bot_frame, text='Confirm', command=self.import_model)
        self.model_btn.grid(row=0, column=2, sticky='nsew')

        self.folder_label = tk.Label(self.bot_frame, text="Folder:")
        self.folder_label.grid(row=1, column=0, sticky='nsew')
        self.folder_entry = tk.Entry(self.bot_frame, textvariable=self.foldername)
        self.folder_entry.insert(0, '/home/gu/fluently_ws/fluently_mem/data/pics_18650')
        self.folder_entry.grid(row=1, column=1, sticky='nsew')
        self.folder_btn = tk.Button(self.bot_frame, text='Confirm', command=self.save_foldername)
        self.folder_btn.grid(row=1, column=2, sticky='nsew')
        
        self.add_bb_btn = tk.Button(self.bot_frame, text='add', command=self.add_bb)
        self.add_bb_btn.grid(row=2, column=0, sticky='nsew')
        self.next_btn = tk.Button(self.bot_frame, text='next', command=self.next)
        self.next_btn.grid(row=2, column=1, columnspan=2, sticky='nsew')

    def save_foldername(self):
        self.idx = 0
        self.foldername = self.folder_entry.get()

    def import_model(self):
        self.model = YOLO(self.model_entry.get())

    def add_bb(self):
        x, y = self.home_frame.canvas.winfo_width() // 2, self.home_frame.canvas.winfo_height() // 2
        self.bb_drawer.add_bb([x-50, y-50, x+50, y+50])

    def classify_and_draw(self, img):
        result = self.model.predict(img, verbose=False)
        drawing_bbs = []
        for i, box in enumerate(result[0].boxes):
            model = box.cls
            confidence = (box.conf)
            x, y, w, h = map(int, box.xywh[0].cpu().numpy())
            drawing_bbs.append([x-w//2, y-w//2, x+w//2, y+w//2])
        self.bb_drawer.add_bbs(drawing_bbs)

    def write_bbs(self):
        with open(os.path.join(self.foldername, 'label', f'pic{self.idx:02d}.txt'))
        

    def next(self):
        self.write_bbs()
        self.idx += 1
        self.bb_drawer = _BoundingBoxEditor(self.home_frame.canvas, self.home_frame)
        self.img = cv2.imread(os.path.join(self.foldername, f'pic{self.idx:02d}.png'))
        self.classify_and_draw(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        self.img = PIL.Image.fromarray(self.img)

    def after_update(self):
        scale, padx, pady = self.home_frame.draw_image(self.img)
        self.bb_drawer.draw_boxes(scale=scale, padx=padx, pady=pady)
        print(self.bb_drawer.bbs_position)
        self.after(1, self.after_update)

class HomeScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.canvas = tk.Canvas(self, bg="#73caec")
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=(0, 0), pady=(0, 0))
    
    def draw_image(self, img):
        scale = min(self.canvas.winfo_width() / img.size[0], self.canvas.winfo_height() / img.size[1])
        padx, pady = 0, 0
        if scale > .01:
            new_size = (int(scale * img.size[0]), int(scale * img.size[1]))
            resized_img = img.resize(new_size)
            padx = (self.canvas.winfo_width() - new_size[0]) // 2
            pady = (self.canvas.winfo_height() - new_size[1]) // 2
        else:
            resized_img = img
        self.tk_image = PIL.ImageTk.PhotoImage(resized_img)
        self.canvas.delete('image')
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, anchor=tk.CENTER, image=self.tk_image, tags='image')
        self.canvas.lower('image')
        return scale, padx, pady

if __name__ == "__main__":
    app = MemGui()
    app.after(1, app.after_update)
    app.mainloop()
