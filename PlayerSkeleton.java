import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public class PlayerSkeleton {

    // Most optimal weights so far
    double[] BEST_PARTICLE = {9.11, -3.56, 4.15, -8.58, -0.60, 5.42, -6.65, 0.77, -3.52, -0.05, -7.60, -1.98, -1.81, -4.23, 4.34};

	public static final Random randomGenerator = new Random();

	// Swarm details

	private static final int SWARM_SIZE = 100;
	private static final int GENERATION_COUNT = 10000;

	// Internal PSO parameters. Sort of a temperature control mechanism.
	private static final double W_HI = 1.0;
	private static final double W_LO = 0.0;
	public static final double C_P = 0.9; // How much weight we want to give personal best.
	public static final double C_G = 0.9; // How much weight we want to give global best.

	private Particle[] swarm = new Particle[SWARM_SIZE];
	
	private double gBest = -1.0;
	private double[] gBestLoc = new double[Particle.FEATURES_COUNT];

	public static void main(String[] args) {
		PlayerSkeleton pso = new PlayerSkeleton();
		pso.run();
	}

	public PlayerSkeleton() {
		randomizeSwarm();
	}

	public void run() {
		while (true) {
			for (int i = 0; i < GENERATION_COUNT; i++) {
				updateSwarmFitness();
				printBest(i);

				double w = calculateSwarmEntropy(i);
				step(w);
				declutterSwarmIfNecessary();
			}
			declutterSwarmIfNecessary();
		}
	}

	/**
	* Re-calculates fitness for all members in the swarm.
	* Also updates personal bests and global best.
	*/
	public void updateSwarmFitness() {
		for (Particle particle : swarm) {
			// Re-evaluate fitness
			particle.fitness = Particle.evaluate(particle.location);
			// Personal best
			particle.updatePersonalBest();
			// Global best
			if (particle.fitness > gBest) {
				gBest = particle.fitness;
				System.arraycopy(particle.location, 0, gBestLoc, 0, Particle.FEATURES_COUNT);
			}
		}
	}

	/**
	* Inertia slowly gains less significance. Point for potential customization.
	*/
	public double calculateSwarmEntropy(int currGeneration) {
		return W_HI - (((double) currGeneration) / GENERATION_COUNT) * (W_HI - W_LO);
	}

	/**
	* Key point in com.cs3243.strategies.pso.PSO. Moves all particles to the next state.
	*/
	public void step(double w) {
		for (Particle particle : swarm) {
			particle.step(w, gBestLoc);
		}
	}

	/**
	* For diagnostic purposes. Prints the gBest so far.
	*
	* @param i: Number to be prepended to the particle info. Generally describes generation state.
	*/
	public void printBest(int i) {
		System.out.printf("%d: %.2f [", i, gBest);
		for (int j = 0; j < gBestLoc.length; j++) {
			System.out.printf(" %.2f", gBestLoc[j]);
		}
		System.out.println(" ]");
	}

	/**
	* Randomly generates a new swarm.
	*/
	public void randomizeSwarm() {
		for (int i = 0; i < SWARM_SIZE; i++) {
			swarm[i] = randomParticle();
		}
	}

	/**
	* Randomly creates a new particle. Used by randomizeSwarm only.
	*/
	public Particle randomParticle() {
		//double[] weights = new double[Particle.FEATURES_COUNT];
		double[] weights = {6.70, -2.63, 6.81, -6.97, -0.44, 4.57, -6.10, -6.74, -6.38, -0.57, -10.00, -6.14, -2.87, -1.88, -1.64};
		double[] velocity = new double[Particle.FEATURES_COUNT];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = (randomGenerator.nextDouble() * (Particle.MAX_POS * 2)) - Particle.MAX_POS;
		}
		for (int i = 0; i < velocity.length; i++) {
			velocity[i] = (randomGenerator.nextDouble() * (Particle.MAX_V * 2)) - Particle.MAX_V;
		}
		return new Particle(weights, velocity);
	}

	private void declutterSwarmIfNecessary() {
		for (Particle particle : swarm)
			for (double vel : particle.velocity)
				if (Math.abs(vel) > 0.1) return;
		System.out.println("declutter");
		randomizeSwarm();
	}


	public static class Particle {
		private static final ExecutorService executor = Executors.newFixedThreadPool(10);

		public static final int FEATURES_COUNT = 15;

		public static final double MAX_V = 5.0;
		public static final double MIN_V = -5.0;
		public static final double MAX_POS = 10.0;
		public static final double MIN_POS = -10.0;

		public double[] location;
		public double[] velocity = new double[FEATURES_COUNT];
		public double fitness = -1.0;

		public double[] pBestLoc = new double[FEATURES_COUNT];
		public double pBestFitness = -1.0;

		public Particle(double[] weights, double[] velocity) {
			this.location = weights; // Not necessary to calculate fitness as this is done by com.cs3243.strategies.pso.PSO.
			this.velocity = velocity;
			this.pBestLoc = weights; // First location is best location so far, personally.
		}

		/**
		 * Assumes fitness has been re-calculated, but personal best has not been updated yet.
		 */
		public void updatePersonalBest() {
			if (fitness > pBestFitness) {
				pBestFitness = fitness;
				System.arraycopy(location, 0, pBestLoc, 0, FEATURES_COUNT);
			}
		}

		/**
		 * Steps to the next generation of the algorithm. Basically updates own velocity and location
		 * based on inertia, personal best and global best.
		 *
		 * @param w: Inertia, is between 0.0 to 1.0, used as multiplier to previous-generation velocity.
		 * @param gBestLoc: Best global location of the previous generation.
		 */
		public void step(double w, double[] gBestLoc) {
			for (int i = 0; i < FEATURES_COUNT; i++) { // Each i corresponds to a dimension.
				double r1 = PlayerSkeleton.randomGenerator.nextDouble();
				double r2 = PlayerSkeleton.randomGenerator.nextDouble();
				double rw = PlayerSkeleton.randomGenerator.nextDouble();
				// First update velocity.
				double inertialVelocity = rw * 2 * velocity[i];
				double personalBestTendency = r1 * PlayerSkeleton.C_P * (pBestLoc[i] - location[i]);
				double globalBestTendency = r2 * PlayerSkeleton.C_G * (gBestLoc[i] - location[i]);
				velocity[i] = inertialVelocity + personalBestTendency + globalBestTendency;
				// Then update location.
				location[i] += velocity[i];
			}

			boundsCheck();
		}

		public void boundsCheck() {
			for (int i = 0; i < FEATURES_COUNT; i++) { // Each i corresponds to a dimension.
				if (velocity[i] > MAX_V) {
					velocity[i] = MAX_V;
				} else if (velocity[i] < MIN_V) {
					velocity[i] = MIN_V;
				}

				if (location[i] > MAX_POS) {
					location[i] = MAX_POS;
					velocity[i] = -velocity[i];
				} else if (location[i] < MIN_POS) {
					location[i] = MIN_POS;
					velocity[i] = -velocity[i];
				}
			}
		}

		public static double evaluate(double[] weights) {
			final GeneticAlgorithm player = new GeneticAlgorithm(new Context(new GeneralMove(weights)));

		/*
		List<CompletableFuture<Integer>> gametests = new ArrayList<>();
		for (int i = 0; i < 10; i++) {
			CompletableFuture<Integer> gametest = new CompletableFuture<>();
			gametests.add(gametest);
			executor.submit(() -> {
				State gameState = new State();
				//PlayerSkeleton.shortRun(gameState, player);
				PlayerSkeleton.headlessRun(gameState, player);
				gametest.complete(gameState.getRowsCleared());
			});
        }

		try {
			// .get() waits for the parent CompletableFuture to finish processing.
			CompletableFuture.allOf(gametests.toArray(new CompletableFuture[10])).get();
		} catch (InterruptedException | ExecutionException e) {
			System.out.println("Thread exception encountered!");
		}

		return gametests.stream().map(gametest -> {
			try {
				return gametest.get();
			} catch (InterruptedException | ExecutionException e) {
				System.out.println("Thread exception encountered!");
			}
			return 0;
		}).reduce(0, (a, b) -> a + b) / 10.0;*/
			//List<Future<Integer>> gametests = new ArrayList<>();
			List<Callable<Integer>> gametests = new ArrayList<>();
			for (int i = 0; i < 10; i++) {
			/*gametests.add(executor.submit(new Callable<Integer>() {
				public Integer call() {
					State gameState = new State();
					PlayerSkeleton.headlessRun(gameState, player);
					return gameState.getRowsCleared();
				}
			}));*/
				gametests.add(new Callable<Integer>() {
					public Integer call() {
						State gameState = new State();
						GeneticAlgorithm.headlessRun(gameState, player);
						return gameState.getRowsCleared();
					}
				});
			}
			try {
				List<Future<Integer>> results = executor.invokeAll(gametests);
				double total = 0;
				for (Future<Integer> result : results) {
					total += result.get();
				}
				return total / results.size();
			} catch (InterruptedException | ExecutionException e) {
				System.out.println("Thread exception encountered!");
			}
			return 0;
		}
	}

	public static class GeneralMove implements MoveStrategy {

		double[] weights;
		Context context;

		public GeneralMove(double[] weights) {
			this.weights = weights;
		}

		@Override
		public int generateMove(State s, int[][] legalMoves) {
			// for all legal moves, choose move with highest fitness function
			int chosen = 0;
			double max_score = Integer.MIN_VALUE;
			for (int i = 0; i < legalMoves.length; i++) {
				// parallelizable
				double score = fitness(new LookaheadState(s), i);
				if (score > max_score) {
					chosen = i;
					max_score = score;
				}
			}

			return chosen;
		}

		@Override
		public void setContext(Context ctx) {
			this.context = ctx;
		}

		public double fitness(LookaheadState s, int move) {
			int completedRows = s.getRowsCleared();
			int premoveHeight = aggregateHeight(s);
			s.makeMove(move);
			if (s.hasLost()) {
				return (double)Integer.MIN_VALUE;
			}
        /*int[] grades = {
                aggregateHeight(s),
                completeLines(s, completedRows),
                holes(s),
                bumpiness(s),
                maxMinDiff(s),
                columnsWithHoles(s),
                rowsWithHoles(s),
                negativeTallestColumnHeight(s)
        };*/
			double[] grades = {
					holes(s),
					//(aggregateHeight(s) - premoveHeight) / State.COLS,
					completeLines(s, completedRows),
					bumpiness(s),
					s.getWalls(),
					s.getVariation(),
					s.getTop()[0],
					s.getTop()[1],
					s.getTop()[2],
					s.getTop()[3],
					s.getTop()[4],
					s.getTop()[5],
					s.getTop()[6],
					s.getTop()[7],
					s.getTop()[8],
					s.getTop()[9]
			};
			return weightFeatures(grades);
		}


		// Grading heuristics
		// Aggregate Height
		// Sum of heights of all columns
		// Goal: MINIMIZE
		private int aggregateHeight(LookaheadState s) {
			int height = 0;
			for (int i = 0; i < s.COLS; i++) {
				height += s.getTop()[i];
			}
			return -height;
		}

		// Complete lines
		// Goal: MAXIMIZE
		private int completeLines(LookaheadState s, int current) {
			return s.getRowsCleared() - current;
		}

		// Holes (squared)
		// Goal: MINIMIZE
		private int holes(LookaheadState s) {
			int holeCount = 0;
			for (int col = 0; col < s.COLS; col++) {
				holeCount += columnHoleCount(s, col);
			}
			return -(holeCount * holeCount);
		}

		private int columnHoleCount(LookaheadState s, int col) {
			int row = 0;
			int count = 0;
			int topRow = s.getTop()[col];
			while (row < topRow) {
				if (s.getField()[row][col] == 0) {
					count += 1;
				}
				row += 1;
			}
			return count;
		}

		// Bumpiness (squared)
		// Goal: MINIMIZE
		private int bumpiness(LookaheadState s) {
			int sumOfHeightDifferences = 0;
			for (int i = 0; i < s.COLS - 1; i++) {
				double difference = Math.pow(s.getTop()[i] - s.getTop()[i + 1], 2);
				sumOfHeightDifferences += difference;
			}
			return -sumOfHeightDifferences;
			//return 0;
		}

		// Difference in height between highest and lowest column
		// Goal: MINIMIZE
		private int maxMinDiff(LookaheadState s) {
			return -Math.abs(tallestColumnHeight(s) - shortestColumnHeight(s));
		}

		// Height of tallest column (SQUARED)
		// Goal: MINIMIZE
		private int negativeTallestColumnHeight(LookaheadState s){
			int height = tallestColumnHeight(s);
			return -(height * height);
		}

		// Height of tallest column
		// Goal: MINIMIZE
		private int tallestColumnHeight(LookaheadState s) {
			int max = 0;
			for (int i = 0; i < s.COLS; i++) {
				int height = s.getTop()[i];
				if (height > max) {
					max = height;
				}
			}

			return max;
		}

		private int shortestColumnHeight(LookaheadState s) {
			int min = s.ROWS;
			for (int i = 0; i < s.COLS; i++) {
				int height = s.getTop()[i];
				if (height < min) {
					min = height;
				}
			}

			return min;
		}

		// The number of columns with at least one hole
		// Goal: MINIMIZE
		private int columnsWithHoles(LookaheadState s) {
			int columns = 0;
			for (int col = 0; col < s.COLS; col++) {
				boolean holeFound = false;
				for (int row = 0; row < s.getTop()[col]; row++) {
					if (holeFound) {
						break;
					}
					if (s.getField()[row][col] == 0) {
						holeFound = true;
					}
				}
				if (holeFound) {
					columns += 1;
				}
			}
			return columns;
		}

		// The number of rows with at least one hole.
		// Goal: MINIMIZE
		private int rowsWithHoles(LookaheadState s) {
			int rows = 0;
			for (int row = 0; row < s.ROWS; row++) {
				boolean holeFound = true;
				for (int col = 0; col < s.COLS; col++) {
					if (holeFound) {
						break;
					}
					if (s.getField()[row][col] == 0 && row < s.getTop()[col]) {
						holeFound = true;
					}
				}
				if (holeFound) {
					rows += 1;
				}
			}
			return rows;
		}

		/**
		 * Applies weight multipliers to grades, and then sum them together to get the final fitness.
		 */
		public double weightFeatures(double[] features) {
			double weightedGrade = 0.0;
			for (int i = 0; i < features.length; i++) weightedGrade += features[i] * weights[i];
			return weightedGrade;
		}
	}

	public class Chromosome {

		private double[] values;
		private int score;

		public Chromosome(double[] values) {
			this.values = values;
			this.score = Integer.MIN_VALUE;
		}

		public int getLength() {
			return this.values.length;
		}

		public void setValue(int index, double value) {
			this.values[index] = value;
		}

		public double getValue(int index) {
			return this.values[index];
		}

		public void setScore(int score) { this.score = score; }

		public int getScore() { return this.score; }

		public double fitness(LookaheadState s, int move) {
			int completedRows = s.getRowsCleared();
			s.makeMove(move);
			int[] grades = {
					aggregateHeight(s),
					completeLines(s, completedRows),
					holes(s),
					bumpiness(s),
					maxMinDiff(s),
					columnsWithHoles(s),
					rowsWithHoles(s),
					tallestColumnHeight(s),
					(int) s.getVariation(),
					s.getWalls(),
					s.getTop()[0],
					s.getTop()[1],
					s.getTop()[2],
					s.getTop()[3],
					s.getTop()[4],
					s.getTop()[5],
					s.getTop()[6],
					s.getTop()[7],
					s.getTop()[8],
					s.getTop()[9],

			};
			return aggregate(grades);
		}

		// Linear summation with weights
		private double aggregate(int[] grades) {
			// Should have the same number.
			assert(grades.length == this.getLength());

			double[] weightedGrades = new double[grades.length];
			double aggregateSum = 0.0;

			for (int i = 0; i < this.getLength(); i++) {
				weightedGrades[i] = (double)grades[i] * this.getValue(i);
				aggregateSum += weightedGrades[i];
			}

			return aggregateSum;
		}

		// Genetic crossover
		public Chromosome crossoverFrom(Chromosome other) {
			assert(this.getLength() == other.getLength());

			Random rand = new Random();
			double[] newValues = new double[this.getLength()];

			for (int i = 0; i < this.getLength(); i++) {
				if (rand.nextFloat() > 0.5) {
					newValues[i] = this.getValue(i);
				} else {
					newValues[i] = other.getValue(i);
				}
			}

			return new Chromosome(newValues);
		}

		// Random mutation
		public void mutate(double limit) {
			assert(limit >= 0.0 && limit <= 1.0);
			Random rand = new Random();
			if (rand.nextFloat() < limit) {
				int idx = rand.nextInt(getLength());
				if (rand.nextFloat() > 0.5) {
					setValue(idx, rand.nextFloat() * -1);
				} else {
					setValue(idx, rand.nextFloat());
				}
			}
		}

		@Override
		public String toString() {
			String s = "";
			for (int i = 0; i < getLength(); i++) {
				s += getValue(i);
				s += ", ";
			}
			return s;
		}

		// Grading heuristics
		// Aggregate Height
		// Sum of heights of all columns
		// Goal: MINIMIZE
		private int aggregateHeight(LookaheadState s) {
			int height = 0;
			for (int i = 0; i < s.COLS; i++) {
				height += s.getTop()[i];
			}
			return height;
		}

		// Complete lines
		// Goal: MAXIMIZE
		private int completeLines(LookaheadState s, int current) {
			return s.getRowsCleared() - current;
		}

		// Holes (squared)
		// Goal: MINIMIZE
		private int holes(LookaheadState s) {
			int holeCount = 0;
			for (int col = 0; col < s.COLS; col++) {
				holeCount += columnHoleCount(s, col);
			}
			return (holeCount * holeCount);
		}

		private int columnHoleCount(LookaheadState s, int col) {
			int row = 0;
			int count = 0;
			int topRow = s.getTop()[col];
			while (row < topRow) {
				if (s.getField()[row][col] == 0) {
					count += 1;
				}
				row += 1;
			}
			return count;
		}

		// Bumpiness
		// Goal: MINIMIZE
		private int bumpiness(LookaheadState s) {
			int sumOfHeightDifferences = 0;
			for (int i = 0; i < s.COLS - 1; i++) {
				double difference = Math.abs(s.getTop()[i] - s.getTop()[i + 1]);
				sumOfHeightDifferences += difference;
			}
			return sumOfHeightDifferences;
		}

		// Difference in height between highest and lowest column
		// Goal: MINIMIZE
		private int maxMinDiff(LookaheadState s) {
			return Math.abs(tallestColumnHeight(s) - shortestColumnHeight(s));
		}

		// Height of tallest column
		// Goal: MINIMIZE
		private int tallestColumnHeight(LookaheadState s) {
			int max = 0;
			for (int i = 0; i < s.COLS; i++) {
				int height = s.getTop()[i];
				if (height > max) {
					max = height;
				}
			}

			return max;
		}

		private int shortestColumnHeight(LookaheadState s) {
			int min = s.ROWS;
			for (int i = 0; i < s.COLS; i++) {
				int height = s.getTop()[i];
				if (height < min) {
					min = height;
				}
			}

			return min;
		}

		// The number of columns with at least one hole
		// Goal: MINIMIZE
		private int columnsWithHoles(LookaheadState s) {
			int columns = 0;
			for (int col = 0; col < s.COLS; col++) {
				boolean holeFound = false;
				for (int row = 0; row < s.getTop()[col]; row++) {
					if (holeFound) {
						break;
					}
					if (s.getField()[row][col] == 0) {
						holeFound = true;
					}
				}
				if (holeFound) {
					columns += 1;
				}
			}
			return columns;
		}

		// The number of rows with at least one hole.
		// Goal: MINIMIZE
		private int rowsWithHoles(LookaheadState s) {
			int rows = 0;
			for (int row = 0; row < s.ROWS; row++) {
				boolean holeFound = true;
				for (int col = 0; col < s.COLS; col++) {
					if (holeFound) {
						break;
					}
					if (s.getField()[row][col] == 0 && row < s.getTop()[col]) {
						holeFound = true;
					}
				}
				if (holeFound) {
					rows += 1;
				}
			}
			return rows;
		}

	}

	public static class GeneticAlgorithm {

		private Context context;

		public GeneticAlgorithm(Context context) {
			this.context = context;
		}

		public int pickMove(State s, int[][] legalMoves) {
			int nextMoveIndex = this.context.executeStrategy(s, legalMoves);
			return nextMoveIndex;
		}

		public static int headlessRun(State s, GeneticAlgorithm p) {
			while(!s.hasLost()) {
				s.makeMove(p.pickMove(s, s.legalMoves()));
			}
			return s.getRowsCleared();
		}

	}

	public static class LookaheadState {
		public static final int COLS = 10;
		public static final int ROWS = 21;
		public static final int N_PIECES = 7;

		public boolean lost = false;

		public TLabel label;

		//current turn
		private int turn = 0;
		private int cleared = 0;

		//each square in the grid - int means empty - other values mean the turn it was placed
		private int[][] field = null;
		//top row+1 of each column
		//0 means empty
		private int[] top = new int[COLS];

		//number of next piece
		protected int nextPiece;

		//all legal moves - first index is piece type - then a list of 2-length arrays
		protected static int[][][] legalMoves = new int[N_PIECES][][];

		//indices for legalMoves
		public static final int ORIENT = 0;
		public static final int SLOT = 1;

		//possible orientations for a given piece type
		protected static int[] pOrients = {1,2,4,4,4,2,2};

		//the next several arrays define the piece vocabulary in detail
		//width of the pieces [piece ID][orientation]
		protected static int[][] pWidth = {
				{2},
				{1,4},
				{2,3,2,3},
				{2,3,2,3},
				{2,3,2,3},
				{3,2},
				{3,2}
		};

		//height of the pieces [piece ID][orientation]
		private static int[][] pHeight = {
				{2},
				{4,1},
				{3,2,3,2},
				{3,2,3,2},
				{3,2,3,2},
				{2,3},
				{2,3}
		};
		private static int[][][] pBottom = {
				{{0,0}},
				{{0},{0,0,0,0}},
				{{0,0},{0,1,1},{2,0},{0,0,0}},
				{{0,0},{0,0,0},{0,2},{1,1,0}},
				{{0,1},{1,0,1},{1,0},{0,0,0}},
				{{0,0,1},{1,0}},
				{{1,0,0},{0,1}}
		};
		private static int[][][] pTop = {
				{{2,2}},
				{{4},{1,1,1,1}},
				{{3,1},{2,2,2},{3,3},{1,1,2}},
				{{1,3},{2,1,1},{3,3},{2,2,2}},
				{{3,2},{2,2,2},{2,3},{1,2,1}},
				{{1,2,2},{3,2}},
				{{2,2,1},{2,3}}
		};

		//initialize legalMoves
		private void initLegalMoves() {
			//for each piece type
			for(int i = 0; i < N_PIECES; i++) {
				//figure number of legal moves
				int n = 0;
				for(int j = 0; j < pOrients[i]; j++) {
					//number of locations in this orientation
					n += COLS+1-pWidth[i][j];
				}
				//allocate space
				legalMoves[i] = new int[n][2];
				//for each orientation
				n = 0;
				for(int j = 0; j < pOrients[i]; j++) {
					//for each slot
					for(int k = 0; k < COLS+1-pWidth[i][j];k++) {
						legalMoves[i][n][ORIENT] = j;
						legalMoves[i][n][SLOT] = k;
						n++;
					}
				}
			}
		}

		public int[][] getField() {
			return field;
		}

		public int[] getTop() {
			return top;
		}

		public static int[] getpOrients() {
			return pOrients;
		}

		public static int[][] getpWidth() {
			return pWidth;
		}

		public static int[][] getpHeight() {
			return pHeight;
		}

		public static int[][][] getpBottom() {
			return pBottom;
		}

		public static int[][][] getpTop() {
			return pTop;
		}

		public int getNextPiece() {
			return nextPiece;
		}

		public boolean hasLost() {
			return lost;
		}

		public int getRowsCleared() {
			return cleared;
		}

		public int getTurnNumber() {
			return turn;
		}

		//constructor
		public LookaheadState() {
			field = new int[ROWS][COLS];
			initLegalMoves();
			nextPiece = randomPiece();
		}

		// copy constructor
		public LookaheadState(State s) {
			this.lost = s.hasLost();
			this.label = s.label;
			this.turn = s.getTurnNumber();
			this.cleared = s.getRowsCleared();
			this.nextPiece = s.getNextPiece();
			this.legalMoves = s.legalMoves;

			this.top = s.getTop().clone();
			this.field = new int[ROWS][];

			int[][] field = s.getField();

			for (int i = field.length - 1; i >= 0 ; i--) {
				this.field[i] = new int[field[i].length];
				for (int j = field[i].length - 1; j >= 0; j--) {
					this.field[i][j] = field[i][j];
				}
			}
		}

		public LookaheadState(LookaheadState s) {
			this.lost = s.hasLost();
			this.label = s.label;
			this.turn = s.getTurnNumber();
			this.cleared = s.getRowsCleared();
			this.nextPiece = s.getNextPiece();
			this.legalMoves = s.getAllLegalMoves();

			this.top = s.getTop().clone();
			this.field = new int[ROWS][];

			int[][] field = s.getField();

			for (int i = field.length - 1; i >= 0 ; i--) {
				this.field[i] = new int[field[i].length];
				for (int j = field[i].length - 1; j >= 0; j--) {
					this.field[i][j] = field[i][j];
				}
			}
		}

		//random integer, returns 0-6
		private int randomPiece() {
			return (int)(Math.random()*N_PIECES);
		}

		//gives legal moves for
		public int[][] legalMoves() {
			return legalMoves[nextPiece];
		}

		public int[][][] getAllLegalMoves() { return legalMoves; }

		//make a move based on the move index - its order in the legalMoves list
		public void makeMove(int move) {
			makeMove(legalMoves[nextPiece][move]);
		}

		//make a move based on an array of orient and slot
		public void makeMove(int[] move) {
			makeMove(move[ORIENT],move[SLOT]);
		}

		//returns false if you lose - true otherwise
		public boolean makeMove(int orient, int slot) {
			turn++;
			//height if the first column makes contact
			int height = top[slot]-pBottom[nextPiece][orient][0];
			//for each column beyond the first in the piece
			for(int c = 1; c < pWidth[nextPiece][orient];c++) {
				height = Math.max(height,top[slot+c]-pBottom[nextPiece][orient][c]);
			}

			//check if game ended
			if(height+pHeight[nextPiece][orient] >= ROWS) {
				lost = true;
				return false;
			}


			//for each column in the piece - fill in the appropriate blocks
			for(int i = 0; i < pWidth[nextPiece][orient]; i++) {

				//from bottom to top of brick
				for(int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
					field[h][i+slot] = turn;
				}
			}

			//adjust top
			for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
				top[slot+c]=height+pTop[nextPiece][orient][c];
			}

			int rowsCleared = 0;

			//check for full rows - starting at the top
			for(int r = height+pHeight[nextPiece][orient]-1; r >= height; r--) {
				//check all columns in the row
				boolean full = true;
				for(int c = 0; c < COLS; c++) {
					if(field[r][c] == 0) {
						full = false;
						break;
					}
				}
				//if the row was full - remove it and slide above stuff down
				if(full) {
					rowsCleared++;
					cleared++;
					//for each column
					for(int c = 0; c < COLS; c++) {

						//slide down all bricks
						for(int i = r; i < top[c]; i++) {
							field[i][c] = field[i+1][c];
						}
						//lower the top
						top[c]--;
						while(top[c]>=1 && field[top[c]-1][c]==0)	top[c]--;
					}
				}
			}

			//pick a new piece
			nextPiece = randomPiece();

			return true;
		}

		public void draw() {
			label.clear();
			label.setPenRadius();
			//outline board
			label.line(0, 0, 0, ROWS+5);
			label.line(COLS, 0, COLS, ROWS+5);
			label.line(0, 0, COLS, 0);
			label.line(0, ROWS-1, COLS, ROWS-1);

			//show bricks

			for(int c = 0; c < COLS; c++) {
				for(int r = 0; r < top[c]; r++) {
					if(field[r][c] != 0) {
						drawBrick(c,r);
					}
				}
			}

			for(int i = 0; i < COLS; i++) {
				label.setPenColor(Color.red);
				label.line(i, top[i], i+1, top[i]);
				label.setPenColor();
			}

			label.show();
		}

		public static final Color brickCol = Color.gray;

		private void drawBrick(int c, int r) {
			label.filledRectangleLL(c, r, 1, 1, brickCol);
			label.rectangleLL(c, r, 1, 1);
		}

		public void drawNext(int slot, int orient) {
			for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
				for(int j = pBottom[nextPiece][orient][i]; j <pTop[nextPiece][orient][i]; j++) {
					drawBrick(i+slot, j+ROWS+1);
				}
			}
			label.show();
		}

		//visualization
		//clears the area where the next piece is shown (top)
		public void clearNext() {
			label.filledRectangleLL(0, ROWS+.9, COLS, 4.2, TLabel.DEFAULT_CLEAR_COLOR);
			label.line(0, 0, 0, ROWS+5);
			label.line(COLS, 0, COLS, ROWS+5);
		}

		// More features
		public double getVariation() {
			double sum = 0;
			double maximumHeight = 0;
			double minimumHeight = 10;
			double mean;
			double variation;
			for (int i = 0; i< 10; i++){
				//featureFactor[i] = tmpState.getColumnHeight(i);
				sum += top[i];
				if (top[i] > maximumHeight){
					maximumHeight = top[i];
				}
				if (top[i] < minimumHeight){
					minimumHeight = top[i];
				}
			}
			mean = sum / 10;
			variation = 0;
			for (int i = 0; i< 10; i++){
				variation += (mean - top[i])*(mean - top[i]);
			}
			return variation;
		}

		public int getWalls(){
			int result = 0;
			for (int i = 1; i < COLS-1; i++ )
				if ((top[i-1] - top[i] >= 2)&&(top[i+1] - top[i] >= 2))
					result += Math.min(top[i-1] - top[i],top[i+1] - top[i]);
			if (top[1] - top[0] >= 2) result += top[1] - top[0];
			if (top[COLS-2] - top[COLS - 1] >= 2) result += top[COLS-2] - top[COLS - 1];
			return result;

		}

		// For lookahead
		public void setNextPiece(int nPiece) {
			nextPiece = nPiece;
		}
	}

	public static interface MoveStrategy {

		public int generateMove(State s, int[][] legalMoves);

		public void setContext(Context ctx);

	}

	public static class Context {

		private MoveStrategy strategy;

		public Context(MoveStrategy strategy) {
			this.strategy = strategy;
			strategy.setContext(this);
		}

		public int executeStrategy(State s, int[][] legalMoves) {
			return this.strategy.generateMove(s, legalMoves);
		}

	}

}
