# Computational-Intelligence
My "Computational Intelligence" course projects at AUT.

<details>
  <summary><h3>Artificial Neural Network</h3></summary>
  This project is a Jupiter Notebook in which a NN is implemented from scratch (only using numpy) to classify the images in the MNIST dataset; where each image is a single digit from 0 to 9.<br><br>
  
  The project's full description (in Persian) can be found [here](https://github.com/NegarMov/Computational-Intelligence/blob/master/ANN_9831062/CI_project1%20-%20Description.pdf).  

  The model is basically a NN that uses feed forward to compute to output and mini-batch gradient descent in the backward phase to update the weights and biases. The gradient can be computed using both <code>compute_grads</code> and <code>compute_grads_vectorized</code> functions, however the <code>compute_grads_vectorized</code> method takes advantage of vectorization and is much faster than the <code>compute_grads</code> method.<br>
  The training accuracy and the test accuracy of the model are both around 94% after only 5 epochs.
</details>

<details>
  <summary><h3>Evolutionary Games</h3></summary>
  In this project, the agent needs to learn how to play a simple 2D game which can be run in one of the 3 following modes.
  <br><br>

  Helicopter             |  Gravity          |  Thrust
  :-------------------------:|:-------------------------:|:-------------------------:
  ![Helicopter](/EvolutionaryGames_9831062/screenshots/helicopter.png?raw=true)  |  ![Gravity](/EvolutionaryGames_9831062/screenshots/gravity.png?raw=true) | ![Thrust](/EvolutionaryGames_9831062/screenshots/thrust.png?raw=true)

  The project's full description (in Persian) can be found [here](https://github.com/NegarMov/Computational-Intelligence/blob/master/EvolutionaryGames_9831062/Evolution%20Project%20-%20Description.pdf).  

  It uses a neural network to compute the output at each moment based on some input features (such as the relative distance to the 2 closest columns) and uses the genetics algorithm to update the network's weights and biases. In the implemented genetic algorithm the next generation is chosen by the Roulette Wheel algorithm, the parents are chosen by the Q-tournament algorithm and the children are generated using crossover and mutation (each by an arbitrary chance).

  You can use the following command to run the program:<br>
  <code>python game.py --mode $mode$ --checkpoint checkpoint/$mode$/$gen_num$</code><br>
  Where <code>mode</code> is one of the 3 modes above and <code>gen_num</code> in the optional <code>--checkpoint</code> parameter indicates the generation number to load the program from.<br>
  
  You can also use the following command to plot the minimum, average and maximum fitness of each generation:<br>
  <code>python plot.py</code>
</details>
