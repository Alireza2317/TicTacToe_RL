import sys
import pygame as pg

W: int = 450
H: int = 550

WIDTH: int = 450
HEIGHT: int = WIDTH
CELL_SIZE: int = WIDTH // 3

FONT_SIZE = 30
GRID_COLOR = (200, 200, 200)
BG_COLOR = (20, 20, 20)
GRID_THICKNESS = 5

FPS: float = 20

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

	def check_win(self) -> str | None:
		"""
			Checks the game board and returns one of these:
			'x' if x wins
			'o' if o wins
			None if the board is not completely filled and nobody won either
			'draw' if the board is full and nobody won
		"""
		winning_combos = [
			((0, 0), (0, 1), (0, 2)),
			((1, 0), (1, 1), (1, 2)),
			((2, 0), (2, 1), (2, 2)),
			((0, 0), (1, 0), (2, 0)),
			((0, 1), (1, 1), (2, 1)),
			((0, 2), (1, 2), (2, 2)),
			((2, 0), (1, 1), (0, 2)),
			((0, 0), (1, 1), (2, 2)),
		]

		for combo in winning_combos:
			# first check if all the positions are filled
			# if not, just check the next combo
			if not all(pos in self.squares for pos in combo):
				continue

			# check if all the values are all the same or not
			# get all the 3 values and put them in a set
			values = {self.squares.get(pos) for pos in combo}
			if len(values) == 1:
				return values.pop()
		if len(self.squares) == 9:
			return 'draw'

		return None

	def game_over_screen(self, message) -> None:
		self.screen.fill(BG_COLOR)
		self.draw_grid()
		self.draw_xo()
		pg.mouse.set_cursor(pg.SYSTEM_CURSOR_ARROW)
		
		text = self.font.render(message, True, (220, 0, 220))
		self.screen.blit(text, (10, HEIGHT+20))
		pg.display.update()

		while True:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					sys.exit()


	def step(self) -> bool:
		x, y = pg.mouse.get_pos()
		if x < WIDTH and y < HEIGHT:
			row = y // CELL_SIZE
			col = x // CELL_SIZE
			if (row, col) in self.squares:
				pg.mouse.set_cursor(pg.SYSTEM_CURSOR_ARROW)
			else:
				pg.mouse.set_cursor(pg.SYSTEM_CURSOR_HAND)
		else:
			pg.mouse.set_cursor(pg.SYSTEM_CURSOR_ARROW)
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

		if (status := self.check_win()):
			if status != 'draw':
				self.game_over_screen(f'{status} won!')
			else:
				self.game_over_screen(status)

			# game over
			return True

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