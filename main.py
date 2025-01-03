import sys
import pygame as pg

W: int = 450
H: int = 550

WIDTH: int = 450
HEIGHT: int = WIDTH
CELL_SIZE: int = WIDTH // 3

FONT_SIZE = 24
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
		self.turn = 'o'

		# keep track of squares and the content
		# ignoring clicking the filled squares later on
		self.squares: dict[tuple[int, int], str] = {}

		# draw the game's grid
		self.draw_grid()

	def draw_grid(self) -> None:
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

	def draw_xo(self) -> None:
		for (row, col), mark in self.squares.items():
			center_y = (row * CELL_SIZE) + (CELL_SIZE // 2)
			center_x = (col * CELL_SIZE) + (CELL_SIZE // 2)

			if mark == 'o':
				pg.draw.circle(
					self.screen,
					color=(200, 0, 0),
					center=(center_x, center_y),
					radius=CELL_SIZE//5,
					width=CELL_SIZE//12
				)
			elif mark == 'x':
				margin = CELL_SIZE//6
				pg.draw.line(
					self.screen,
					color=(0, 200, 0),
					start_pos=(center_x-margin, center_y-margin),
					end_pos=(center_x+margin, center_y+margin),
					width=15
				)
				pg.draw.line(
					self.screen,
					color=(0, 200, 0),
					start_pos=(center_x+margin, center_y-margin),
					end_pos=(center_x-margin, center_y+margin),
					width=15
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

				# ignore the click if it is within a filled square
				if (row, col) in self.squares:
					break

				# add the current square location and content
				self.squares.update({(row, col): self.turn})

				# change the turn after the user played
				self.turn = 'o' if self.turn == 'x' else 'x'

		self.screen.fill(BG_COLOR)
		self.draw_grid()
		self.draw_xo()
		text = self.font.render(f'Turn: {self.turn}', False, GRID_COLOR)
		self.screen.blit(text, (10, HEIGHT+20))
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