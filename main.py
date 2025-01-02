import sys
import pygame as pg

W: int = 450
H: int = 550

WIDTH: int = 450
HEIGHT: int = WIDTH
CELL_SIZE: int = WIDTH // 3

FONT_SIZE = 12
GRID_COLOR = (200, 200, 200)
BG_COLOR = (20, 20, 20)
GRID_THICKNESS = 5

FPS: float = 10

class TicTacToeGame:
	def __init__(self) -> None:
		pg.init()
		self.screen = pg.display.set_mode((W, H))
		pg.display.set_caption('Tic Tac Toe')
		self.clock = pg.time.Clock()
		self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)
		self.screen.fill(color=BG_COLOR)

		# turn could be 'x' or 'o'
		self.turn = 'x'

		# draw the game's grid
		# vertical lines
		for i in range(1, 3):
			pg.draw.line(
				surface=self.screen,
				color=GRID_COLOR,
				start_pos=(i * CELL_SIZE, 0),
				end_pos=(i * CELL_SIZE, HEIGHT),
				width=GRID_THICKNESS
			)

		# horizental lines
		for i in range(1, 3):
			pg.draw.line(
				surface=self.screen,
				color=GRID_COLOR,
				start_pos=(0, i * CELL_SIZE),
				end_pos=(WIDTH, i * CELL_SIZE),
				width=GRID_THICKNESS
			)


	def step(self) -> bool:
		# handle user events
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

			if event.type == pg.MOUSEBUTTONDOWN:
				x, y = pg.mouse.get_pos()
				col = x // CELL_SIZE
				row = y // CELL_SIZE
				center_y = (row * CELL_SIZE) + (CELL_SIZE // 2)
				center_x = (col * CELL_SIZE) + (CELL_SIZE // 2)

				if self.turn == 'o':
					pg.draw.circle(
						self.screen,
						color=(200, 0, 0),
						center=(center_x, center_y),
						radius=CELL_SIZE//5,
						width=CELL_SIZE//12
					)
				elif self.turn == 'x':
					pg.draw.circle(
						self.screen,
						color=(0, 200, 0),
						center=(center_x, center_y),
						radius=CELL_SIZE//5,
						width=CELL_SIZE//12
					)

				# change the turn after the user played
				self.turn = 'o' if self.turn == 'x' else 'x'

		pg.display.update()
		self.clock.tick(FPS)
		return False

if __name__ == '__main__':
	game = TicTacToeGame()

	while True:
		game_over = game.step()

		if game_over:
			break

	pg.quit()
	sys.exit()