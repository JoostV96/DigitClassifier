import numpy as np
import pygame
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from tkinter import Tk
from tkinter import messagebox

path = "C:/Users/joost/.spyder-py3/Overig/Sudoku/MNIST_model.h5"
model = tf.keras.models.load_model(
    path,
    custom_objects={'KerasLayer': hub.KerasLayer}
)

DIM = 28  # picture will be 28x28
WIDTH = 560
HEIGHT = 660
CELL_DIM = WIDTH // DIM

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Canvas:
    
    def __init__(self, dim):
        self.canvas = np.zeros((dim,dim))
        
    def insert(self, row, col):
        if row >= 0 and row < DIM and col >= 0 and col < DIM:
            self.canvas[row, col] = 1
    
    def delete(self, row, col):
        self.canvas[row, col] = 0
    
    def reset(self, dim):
        self.canvas = np.zeros((dim, dim))
    
        
class Drawer:
    
    def __init__(self, screen):
        self.screen = screen
    
    def draw_board(self, canvas):
        self.screen.fill(WHITE)
        for row in range(DIM):
            for col in range(DIM):
                if canvas.canvas[row, col] == 1:  # Draw black square
                    pygame.draw.rect(self.screen, BLACK, (col*CELL_DIM, row*CELL_DIM, CELL_DIM, CELL_DIM),0)
    
        self.draw_button(start_x=0, start_y=565, text="DONE", margin=0)
        self.draw_button(start_x=WIDTH-WIDTH//3, start_y=565, text="RESET", margin=0)

    def draw_button(self, start_x, start_y, text, margin):
        """Draws a "button" to the game board."""
        pygame.draw.rect(self.screen, RED, (start_x, start_y, WIDTH//3, 50))
        pygame.draw.rect(self.screen, BLACK, (start_x, start_y, WIDTH//3, 50),3)
        myfont = pygame.font.SysFont(None, 72)
        textsurface = myfont.render(text, False, BLACK)
        pos = (start_x, start_y)
        self.screen.blit(textsurface, pos)
    
    @staticmethod
    def mouse_to_cell(pos_x, pos_y):
        return pos_x // CELL_DIM, pos_y // CELL_DIM
        
    
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('Canvas')

canvas = Canvas(DIM)
drawer = Drawer(screen)

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if pygame.mouse.get_pressed()[0]:
            (pos_x, pos_y) = pygame.mouse.get_pos()   
            col, row = drawer.mouse_to_cell(pos_x, pos_y)   
            if row < DIM:
                canvas.insert(row, col)
                
                canvas.insert(row, col-1)            
                canvas.insert(row, col+1)
                canvas.insert(row-1, col-1)
                canvas.insert(row-1, col)
                canvas.insert(row-1, col+1)
                canvas.insert(row+1, col-1)
                canvas.insert(row+1, col)
                canvas.insert(row+1, col+1)
                
            if pos_x >= 400 and pos_x <= WIDTH and pos_y >= WIDTH:
                canvas.reset(DIM)
            if pos_x >= 0 and pos_x <= WIDTH//3 and pos_y >= WIDTH:
                done = True
                pred = model.predict(canvas.canvas.reshape(1, 28, 28, 1), batch_size=1)  
                Tk().wm_withdraw() #to hide the main window
                messagebox.showinfo('Prediction','You drew a {}!'.format(pred.argmax()))
                break

    drawer.draw_board(canvas)
    pygame.display.update()
pygame.quit()