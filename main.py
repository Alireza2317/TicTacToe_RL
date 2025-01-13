import sys
from random import randint, random, choice
import pygame as pg
import numpy as np
from nn import NeuralNetwork

W: int = 620
H: int = 700

WIDTH: int = W
HEIGHT: int = WIDTH
CELL_SIZE: int = WIDTH // 3

FONT_SIZE = 26
BG_COLOR = '#121212'
GRID_COLOR = '#3a3a3a'
TEXT_COLOR = '#cccccc'
X_COLOR = '#00d1ff'
O_COLOR = '#ff6e6e'
GRID_THICKNESS = 15

# the computer is X or state 1
# the computer will always play the first move
AI_MARK = 'x'

FPS: float = 20

class TicTacToeGame:
	def __init__(self, render_enabled: bool = True) -> None:
		self.render_enabled = render_enabled

		if self.render_enabled:
			pg.init()
			self.screen = pg.display.set_mode((W, H))
			self.screen.fill(color=BG_COLOR)

			pg.display.set_caption('Tic Tac Toe')

			self.clock = pg.time.Clock()
			self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

			self.game_surface = pg.Surface(size=(W, H))
			self.game_surface.fill(color=BG_COLOR)

			self.text_surface = self.game_surface.subsurface((0, HEIGHT, WIDTH, H-HEIGHT))

			self.draw_grid()

		self.reset()

	def reset(self) -> None:
		# turn could be 'x' or 'o'
		self.turn = AI_MARK

		# keep track of squares and the content
		# ignoring clicking the filled squares later on
		self.squares: dict[tuple[int, int], str] = {}

	def render(self) -> None:
		"""
			rendering all the game board and the turn text
		"""
		# drawing all the marks based on self.squares
		self.game_surface.fill(BG_COLOR)
		self.draw_grid()
		self.draw_xo()

		# putting the text on the text surface
		text = self.font.render(f'Turn: {self.turn}', False, TEXT_COLOR)
		self.text_surface.blit(text, (10, 10))

		self.screen.blit(self.game_surface, dest=(0, 0))
		pg.display.update()
		self.clock.tick(FPS)

	def draw_grid(self) -> None:
		# vertical lines
		for i in range(1, 3):
			pg.draw.line(
				surface=self.game_surface,
				color=GRID_COLOR,
				start_pos=(i * CELL_SIZE, 0),
				end_pos=(i * CELL_SIZE, HEIGHT),
				width=GRID_THICKNESS
			)

		# horizental lines
		for i in range(1, 3):
			pg.draw.line(
				surface=self.game_surface,
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
					self.game_surface,
					color=O_COLOR,
					center=(center_x, center_y),
					radius=CELL_SIZE//4.8,
					width=CELL_SIZE//14
				)
			elif mark == 'x':
				margin = CELL_SIZE//6
				pg.draw.line(
					self.game_surface,
					color=X_COLOR,
					start_pos=(center_x-margin, center_y-margin),
					end_pos=(center_x+margin, center_y+margin),
					width=15
				)
				pg.draw.line(
					self.game_surface,
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

	def get_state(self) -> list[int]:
		"""
			Returns the current board state as a flat list(vector) of:
			+1 as in X (AI_MARK),
			-1 as in O (oponent's mark),
			0 as in empty cell
		"""
		state: list[int] = [0 for _ in range(9)]

		for (row, col), mark in self.squares.items():
			state[(row * 3) + col] = 1 if mark == AI_MARK else -1

		return state

	def get_reward(self) -> float | None:
		"""
			Returns the reward for the current game state:
			+1.0 for a win by X
			-1.0 for a win by O
			-0.5 for draw
			None if the game continues
		"""

		game_over: bool | None = self.check_win()

		if not game_over: return None

		match game_over:
			case 'x':
				return +1.0
			case 'o':
				return -1.0
			case 'draw':
				return -0.2

	def switch_turns(self) -> None:
		self.turn = 'o' if self.turn == 'x' else 'x'

	def step(self, action: int) -> tuple[list[int], float, bool]:
		"""
			steps the game with the given action
			only handles the ai's turn and move
			returns the next_state, reward, done(game over flag)
		"""

		if self.turn == AI_MARK:
			# since the agent never chooses an invalid action
			# there is no need to check action's valididity
			coordinate = divmod(action, 3)
		
			# update the board
			self.squares.update({coordinate: AI_MARK})
			
		if self.render_enabled:
			self.render()
		self.switch_turns()

		if (status := self.check_win()):
			match status:
				case 'draw':
					print('Draw.')
				case 'x':
					print('X Wins!')
				case 'o':
					print('O Wins!')
			
			reward = self.get_reward()
			state = self.get_state()

			self.reset()
			if self.render_enabled:
				self.render()
			
			return state, reward, True


		# the game continues
		return self.get_state(), 0, False

	def get_click_coordinate(self) -> tuple[int, int]:
		"""
			getting the user's move from the mouse clicks
			returns the coordinate based on the game's 3x3 board
		"""
		x, y = pg.mouse.get_pos()
		col = x // CELL_SIZE
		row = y // CELL_SIZE

		return (row, col)

class Agent:
	def __init__(self) -> None:
		# creating the neural network
		# 9 inputs, the game board
		# 0: empty cells, 1: my mark, -1: opponent's mark
		# 9 outputs representing Q(s, a) for each cell in the board
		self.network = NeuralNetwork(
			layers_structure=[9, 40, 9],
			activations='tanh'
		)

		# discount factor
		self.gamma: float = 0.9

		# learning rate
		self.alpha: float = 0.002

		# epsilon-greedy policy for explore-exploit trade-off
		# should decay over training to lower the exploration
		self.epsilon: float = 1

	def choose_action(self, state: list[int]) -> int:
		"""
		chooses and returns only a VALID move based on the current state
		"""
		# if the state[i] == 0, then the cell is empty and is a valid move
		valid_actions = [i for i in range(9) if state[i] == 0]
		
		# with probability epsilon, pick a random action
		if random() < self.epsilon:
			return choice(valid_actions)
	
		# otherwise pick the action with the highest Q(a, s)
		else:
			q_values = self.network.predict_output(state)

			return max(valid_actions, key=lambda x: q_values[x])

	def decay_epsilon(self) -> None:
		"""
			decay epsilon over time to minimize exploration
		"""
		self.epsilon = max(0.1, self.epsilon * 0.995)

	def update(
			self,
			state: list[int],
			action: int,
			reward: float,
			next_state: list[int],
			done: bool
	) -> None:
		"""
			updates the neural network using the Bellman equation.

			@param state: the current state of the game
			@param action: the action that the agent picked for the state
			@param reward: the reward that the agent received for the action
			@param next_state: the state after the agent's action
			@param done: if the game is over or not
		"""
		q_values = self.network.predict_output(state)
		target_q_values = q_values.copy()

		if done:
			# game is over
			target_q_values[action] = reward
		else:
			# using Bellman equation to compute the target
			next_q_values = self.network.predict_output(next_state)
			target_q_values[action] = reward + self.gamma * np.max(next_q_values)

		# train the network
		state = np.array(state).reshape((-1, 1))
		target_q_values = np.array(target_q_values).reshape((-1, 1))
		self.network.train(
			x_train=state,
			y_train=target_q_values,
			learning_rate=self.alpha,
			constant_lr=True,
			batch_size=1,
			number_of_epochs=5, 
			verbose=False
		)


def train_agent_manually(resume: bool = False, episodes=20) -> None:
	agent = Agent()
	game = TicTacToeGame(render_enabled=True)

	if resume:
		try:
			with open('params.txt', 'r') as file:
				eps = float(file.read().strip())
				agent.epsilon = eps
			
			agent.network.load_params_from_file('nn_params.txt')
		except FileNotFoundError:
			print('No trained file was found, training from scratch!')
				
	total_reward: float = 0
	for episode in range(1, episodes + 1):
		print(f'Episode {episode}:\t{total_reward=:.1f}, {agent.epsilon=:.3f}')
		
		game.reset()
		state: list[int] = game.get_state()
		done: bool = False

		# this loop will run while the game is running
		while not done:
			if game.turn == AI_MARK:
				# choose an action based on the current state
				ai_action = agent.choose_action(state=state)

				# step the game with the AI's move
				next_state, reward, done = game.step(ai_action)

				# should always update after the human's response
				# unless the ai's move, terminates the game
				if done:
					agent.update(state, ai_action, reward, next_state, done)
					total_reward += reward
			else:
				# handle user's input
				for event in pg.event.get():
					if event.type == pg.QUIT:
						pg.quit()
						sys.exit()

					if event.type == pg.MOUSEBUTTONDOWN:
						position: tuple[int, int] = game.get_click_coordinate()

						# ignore invalid moves
						if position in game.squares:
							break
						
						# valid moves here
						
						# update the board
						game.squares.update({position: game.turn})
						
						next_state, reward, done = game.step(action=-1)			

						# update the agent based on the human's response
						agent.update(state, ai_action, reward, next_state, done)
						total_reward += reward
			
			# transition to the next state
			state = next_state

		agent.decay_epsilon()

	agent.network.save_parameters_to_file('nn_params.txt')
	with open('params.txt', 'w') as file:
		file.write(f'{agent.epsilon}')


def train_agent_randomly(resume: bool = False, episodes=10) -> None:
	agent = Agent()
	game = TicTacToeGame(render_enabled=False)
	
	if resume:
		try:
			with open('params.txt', 'r') as file:
				eps = float(file.read().strip())
				agent.epsilon = eps
			
			agent.network.load_params_from_file('nn_params.txt')
		except FileNotFoundError:
			print('No trained file was found, training from scratch!')
				
	total_reward: float = 0
	for episode in range(1, episodes + 1):
		if episode%10 == 0:
			print(f'Episode {episode}:\t{total_reward=:.1f}, {agent.epsilon=:.3f}')
		
		game.reset()
		state: list[int] = game.get_state()
		done: bool = False

		# this loop will run while the game is running
		while not done:
			if game.turn == AI_MARK:
				# choose an action based on the current state
				ai_action = agent.choose_action(state=state)

				# step the game with the AI's move
				next_state, reward, done = game.step(ai_action)

				# should always update after the human's response
				# unless the ai's move, terminates the game
				if done:
					agent.update(state, ai_action, reward, next_state, done)
					total_reward += reward
			else:
				# choose a valid random move
				# if the state[i] == 0, then the cell is empty and is a valid move
				valid_actions = [i for i in range(9) if state[i] == 0]
		
				# pick a random move
				opponent_action = choice(valid_actions)
				position: tuple[int, int] = divmod(opponent_action, 3)

				# update the board
				game.squares.update({position: game.turn})
				
				next_state, reward, done = game.step(action=-1)			

				# update the agent based on the human's response
				agent.update(state, ai_action, reward, next_state, done)
				total_reward += reward
	
			# transition to the next state
			state = next_state

		agent.decay_epsilon()

	agent.network.save_parameters_to_file('nn_params.txt')
	with open('params.txt', 'w') as file:
		file.write(f'{agent.epsilon}')


def play_with_ai():
	game = TicTacToeGame()
	agent = Agent()
	
	# no exploration
	agent.epsilon = 0
	agent.network.load_params_from_file('nn_params.txt')

	game_over = False

	while not game_over:
		if game.turn == AI_MARK:
			state = game.get_state()
			action = agent.choose_action(state)
			_, _, game_over = game.step(action=action)

		else:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					sys.exit()
				
				if event.type == pg.MOUSEBUTTONDOWN:
					coordinate = game.get_click_coordinate()
					
					if coordinate in game.squares:
						break

					game.squares.update({coordinate: game.turn})
					_, _, game_over = game.step(action=-1)
	

def test_agent(agent: Agent):
	states = [
		[1, 0, 0, 1, 0, -1, 0, 0, -1],
		[-1, -1, 0, 0, 0, 0, 1, 1, 0],
		[0, 0, 0, -1, -1, 1, 1, -1, 1],
		[-1, 0, 0, 1, -1, 0, 1, 0, 0],
		[0, 1, -1, 0, 1, 0, -1, 0, 0],
		[1, -1, -1, -1, 1, 0, 1, 0, 0],
		[1, -1, -1, 0, 0, -1, 0, 1, 1],
		[1, 0, 0, 1, 0, 0, -1, -1, 0],
		[-1, 0, 1, 0, -1, 1, 0, 0, 0],
		[0, 0, 1, 0, 1, -1, 0, 0, -1],
		[0, 1, -1, 0, -1, 1, 0, 0, 0],
		[1, 0, 0, 0, -1, -1, 0, 0, 1],
		[1, 1, 0, 0, 0, 0, 0, -1, -1],
		[-1, 0, 0, 0, 1, 1, -1, 0, 0],
		[0, 0, -1, 1, 1, -1, 0, 0, 0],
		[1, 0, 1, -1, -1, 0, 0, 0, 0],
		[0, -1, 0, 1, 1, 0, 0, -1, 0],
		[-1, 0, 1, 0, 1, 0, 1, 0, -1],
		[0, 1, 0, -1, -1, 0, 0, 1, 0],
		[1, 0, 0, 0, -1, 0, -1, 1, 0],
		[0, 0, 0, 0, -1, -1, 1, 0, 1],
		[-1, 1, 0, -1, 0, 0, 0, 1, 0],
		[1, 0, -1, -1, 0, 0, -1, 1, 1],
		[-1, 0, -1, 1, 1, 0, 0, 0, 0]
	]
	actions = [
		6, 8, 2, 8, 7, 8, 6, 8, 8, 6, 6, 3, 2, 3, 8, 1, 5, 2, 5, 2, 3, 6, 4, 5
	]

	
	states = np.array(states)
	actions = np.array(actions).reshape((-1, 1))

	NUM_SAMPLES = max(actions.shape)
	
	corrects = 0
	for i in range(NUM_SAMPLES):
		prediction = agent.network.predict_class(states[i].reshape((9, 1)))
		if prediction == actions[i]:
			corrects += 1
	
	print(f'Accuracy = {corrects / NUM_SAMPLES * 100:.2f}%')


if __name__ == '__main__':
	#train_agent_manually(resume=True)
	#train_agent_randomly(resume=True)
	#play_with_ai()

	agent = Agent()
	agent.epsilon = 0
	agent.network.load_params_from_file('nn_params.txt')

	test_agent(agent)