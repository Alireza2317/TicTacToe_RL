import sys
import pygame as pg

WIDTH: int = 400
HEIGHT: int = 300
FONT_SIZE = 12

class TicTacToeGame:
	def __init__(self) -> None:
		pg.init()
		self.screen = pg.display.set_mode((WIDTH, HEIGHT))
		pg.display.set_caption('Tic Tac Toe')

		self.clock = pg.time.Clock()

		self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

	def step(self) -> bool:
		# handle user events
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()
			
		return False

if __name__ == '__main__':
	game = TicTacToeGame()

	while True:
		game_over = game.step()

		if game_over:
			break

	pg.quit()
	sys.exit()