import sys
import pygame as pg
from nn import NeuralNetwork

W: int = 450
H: int = 550

WIDTH: int = 450
HEIGHT: int = WIDTH
CELL_SIZE: int = WIDTH // 3

FONT_SIZE = 26
BG_COLOR = '#121212'
GRID_COLOR = '#3a3a3a'
TEXT_COLOR = '#cccccc'
X_COLOR = '#00d1ff'
O_COLOR = '#ff6e6e'
GRID_THICKNESS = 15

FIRST_TURN = 'x'

FPS: float = 20

class TicTacToeGame:
	def __init__(self, render_enabled: bool = True) -> None:
		pg.init()
		self.screen = pg.display.set_mode((W, H))
		self.screen.fill(color=BG_COLOR)

		pg.display.set_caption('Tic Tac Toe')

		self.clock = pg.time.Clock()
		self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

		self.game_surface = pg.Surface(size=(W, H))
		self.game_surface.fill(color=BG_COLOR)
		
		self.render_enabled: bool = render_enabled
		self.reset()

	def reset(self) -> None:
		# turn could be 'x' or 'o'
		self.turn = FIRST_TURN

		# keep track of squares and the content
		# ignoring clicking the filled squares later on
		self.squares: dict[tuple[int, int], str] = {}

	def set_render(self, status: bool = True) -> None:
		self.render_enabled = status

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
					color=O_COLOR,
					center=(center_x, center_y),
					radius=CELL_SIZE//4.8,
					width=CELL_SIZE//14
				)
			elif mark == 'x':
				margin = CELL_SIZE//6
				pg.draw.line(
					self.screen,
					color=X_COLOR,
					start_pos=(center_x-margin, center_y-margin),
					end_pos=(center_x+margin, center_y+margin),
					width=15
				)
				pg.draw.line(
					self.screen,
					color=X_COLOR,
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

	def game_over_screen(self, message) -> bool:
		self.screen.fill(BG_COLOR)
		self.draw_grid()
		self.draw_xo()
		pg.mouse.set_cursor(pg.SYSTEM_CURSOR_ARROW)

		text1 = self.font.render(message, True, TEXT_COLOR)
		text2 = self.font.render('Press R to reset & Q to quit.', True, TEXT_COLOR)
		self.screen.blit(text1, (10, HEIGHT+20))
		self.screen.blit(text2, (10, HEIGHT+50))
		pg.display.update()

		while True:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					sys.exit()

				if event.type == pg.KEYDOWN:
					if event.key == pg.K_r:
						self.reset()
						return False
					if event.key == pg.K_q:
						return True


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


		self.screen.fill(BG_COLOR)
		self.draw_grid()
		self.draw_xo()
		text = self.font.render(f'Turn: {self.turn}', False, TEXT_COLOR)
		self.screen.blit(text, (10, HEIGHT+20))
		pg.display.update()

		self.clock.tick(FPS)

		if (status := self.check_win()):
			match status:
				case 'draw':
					messg = 'Draw.'
				case 'x':
					messg = 'X Wins!'
				case 'o':
					messg = 'O Wins!'

			if not self.game_over_screen(message=messg):
				return False
			return True
		else:
			return False


class Agent:
	def __init(self) -> None:
		# creating the neural network
		# 9 inputs, the game board
		# 0: empty cells, 1: my mark, -1: opponent's mark
		# 9 outputs representing Q(s, a) for each cell in the board
		network = NeuralNetwork(
			layers_structure=[9, 32, 32, 9],
			activations='relu'
		)

		# immediate reward after taking action a in state s
		# 1 for winning
		# -1 for losing
		# -0.5 for draw
		# 0 for ongoing game

		r: float = 0
		# discount factor
		gamma: float = 1




if __name__ == '__main__':
	game = TicTacToeGame()

	while True:
		game_over = game.step()

		if game_over:
			break

	pg.quit()
	sys.exit()