from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import subprocess

class GameGrid(Frame):
    def __init__(self):
        super().__init__()

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            '0': logic.up,
            '1': logic.right,
            '2': logic.down,
            '3': logic.left,
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.update_grid_cells()
        self.mainloop()

    def init_grid(self):
        background = Frame(
            self,
            bg=c.BACKGROUND_COLOR_GAME,
            width=c.SIZE,
            height=c.SIZE
        )
        background.grid()

        cell_width = c.SIZE / c.GRID_LEN
        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=cell_width,
                    height=cell_width
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2
                )
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="",
                        bg=c.BACKGROUND_COLOR_CELL_EMPTY
                    )
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT.get(
                            new_number,
                            c.BACKGROUND_COLOR_CELL_EMPTY
                        ),
                        fg=c.CELL_COLOR_DICT.get(new_number)
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        if key == c.NEXT_BOARD:
            # Write the current matrix to a file
            with open("decision_tree/tree.txt", "w") as f:
                for row in self.matrix:
                    f.write(' '.join(map(str, row)) + '\n')

            # Run the external process to get the next move
            move = subprocess.run(
                ["decision_tree/MCTS"],
                capture_output=True,
                text=True
            ).stdout.strip()

            if move in self.commands:
                self.matrix, done = self.commands[move](self.matrix)
                if done:
                    self.matrix = logic.add_two(self.matrix)
                    self.update_grid_cells()
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.update_grid_cells()

game_grid = GameGrid()
