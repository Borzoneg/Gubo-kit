import tkinter as tk
from tkinter import ttk


class BoundingBoxEditor:
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

    def on_release(self, event):
        """Resets after dragging"""
        self.selected_box = None
        self.dragging = None

class QualitiesEditor:
    def __init__(self, canvas, cell_m_q,  cell_h_q, editable=True):
        self.canvas = canvas
        self.bbs_position = []
        self.keep_bbs = []

        self.m, self.h = cell_m_q, cell_h_q

        self.old_quality = None
        self.editing = False
        self.editing_qual_id = None
        self.edit_entry = tk.Entry(self.canvas, width=4, font=("Arial", 7), justify="center")

        self.boxes = []
        self.editable = editable

        self.canvas.bind("<ButtonPress-1>", self.on_click)

    def lock(self):
        self.editable = False
        self.canvas.unbind("<ButtonPress-1>")

    def add_quals(self, keep_bbs, bbs):
        self.bbs_position = bbs
        self.keep_bbs = keep_bbs

    def clear_quals(self):
        self.bbs_position = []
        self.keep_bbs = []
        self.canvas.delete('quals')
        self.boxes.clear()

    def write_qualities(self, scale=1, padx=0, pady=0):
        self.canvas.delete('quals')
        self.boxes.clear()
        for i, (bb, keep) in enumerate(zip(self.bbs_position, self.keep_bbs)):
            x_min = ((bb[0] * scale) + padx)
            y_min = ((bb[1] * scale) + pady)
            if self.editable:
                self.canvas.create_rectangle(x_min-5, y_min-5, x_min+5, y_min+5, fill="gray20", outline="white", tags='quals') # delete_bg
            if keep:
                txt_box = self.canvas.create_text(x_min, y_min, text="✔", font=("Arial", 10), fill="green2", tag='quals')
            else:
                txt_box = self.canvas.create_text(x_min, y_min, text="✘", font=("Arial", 10), fill="firebrick1", tag='quals')
            self.boxes.append(txt_box)

    def on_click(self, event):
        if self.canvas.find_withtag(tk.CURRENT):  # If clicked on an item
            item = self.canvas.find_withtag(tk.CURRENT)[0]
            for i, label_id in enumerate(self.boxes):
                if label_id == item:
                    self.keep_bbs[i] = not self.keep_bbs[i]