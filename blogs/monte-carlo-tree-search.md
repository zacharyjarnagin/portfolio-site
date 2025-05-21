---
title: 'Monte Carlo Tree Search, Neural Networks, & Chess'
date: '2021-05-01'
lastmod: '2023-11-06'
tags: ['AI', 'School Project']
authors: ['default', 'michaelsteinberg']
draft: false
summary: 'Developing an intelligent chess engine'
bibliographyFilePath: blogs/references/monte-carlo-references.bib
images: ['/static/images/monte-carlo/catfish2_beating_sf_lvl_5.jpeg']
---

_Institution_: Northeastern University, Khoury College of Computer Sciences

_Co-Authored with_ Michael Steinberg

# Abstract

The aim of this project was to develop an agent capable of playing chess intelligently. To achieve this goal, we drew some inspiration from AlphaZero, a world class engine for chess. AlphaZero uses a combination of Monte Carlo Tree Search and Convolution Neural Networks to select the best move. By utilizing MCTS and a CNN, we were able to defeat a random agent without issue and competently battle a Stockfish agent [@stanford]. Based on the claimed Elo of these agents (from the Elo rating system), we can confidently say we developed an intelligent chess engine.

# Introduction

Chess has long been known as one of the gold standards for artificial intelligence problem solving. Typically, a player has around 30 legal moves to make from any given position in chess, and their opponent has roughly 30 possible moves to make in response [@chess]. For a two-ply search, or a white move and black response, there are 900 possibilities to consider [@chess]. A three-ply search, or white move, black response, and white move, consists of 27,000 possibilities to consider [@chess].
Computer agents began competing against human players as early as the 1960s  [@chess]. _MacHack VI_ was a notable program from 1967, losing four games but drawing one in a U.S. Chess Federation tournament  [@chess]. As computer hardware and the field of computer science became more cutting edge over the years, so too did these chess agents  [@chess]. In 1988, a program named _Deep Thought_ managed to defeat grandmaster Ben Larsen [@chess]. Soon, _MacHack VI_ replaced it's predecessor _Deep Thought_ and could now analyze 50 billion positions in just three minutes [@chess].
On May 11th, 1997, history was made by the _MacHack VI_ agent [@kasparov]. It managed to defeat the reining world champion at that time, Garry Kasparov [@kasparov]. _MacHack VI_ played so well that many, Kasparov included, thought _MacHack VI_ was actually controlled by a human grandmaster [@kasparov]. This signalled a pivotal moment in artificial intelligence where an agent could rival the intelligence of a human [@kasparov].
One exceptional modern agent for solving chess is the _Stockfish_ agent, which primarily takes a brute force approach [@stock]. Our goal was to beat an implementation of _Stockfish_ with an implementation of _Monte Carlo Tree Search_ (MCTS), based off of AlphaGo Zero. An adaptation of AlphaGo Zero known as AlphaZero has previously been utilized by Google DeepMind to defeat Stockfish in chess [@agozero]. In this project, we attempted to replicate the performance of that engine.

# Background

## Monte Carlo Tree Search

Monte Carlo Tree Search, or MCTS, is a tree searching algorithm which aims to pick a move that will get the agent closer to winning a zero sum game. It consists primarily of four phases:

1. **Selection:** Here, we use a _tree policy_ to construct a path to a leaf node, a position which ideally has a high probability of winning [@mcts]. These leaf nodes will have unexplored child nodes. We want to maintain a balance between **exploration & exploitation** [@mcts]. Essentially, we want to exploit the move that will get us the closer to winning the game, while also exploring other moves to ensure we don't get tunnel vision and miss on a possibly better move [@mcts].

    We will always choose a child node with the highest _Upper Confidence Bound_ (UCB) [@mcts]. The UCB dictates the highest possible estimation of the strength of a node if we choose this specific node over and over again [@mcts]. This UCB should ensure that the more a node is selected, the less likely it is to be chosen again [@mcts]. This ensures a proper exploration/exploitation trade-off [@mcts].

2. **Expansion:** Once a leaf node is reached, we randomly select one that is unexplored based on the distribution provided by the UCBs [@mcts].

3. **Simulation:** Here is the crux of not only MCTS, but our problem at hand: _the roll-out_ [@mcts]. You will soon see that this step is the main point of our exploration [@mcts]. However, conventional MCTS simply randomly plays out a game, ending in $+1$, $0$, or $-1$ for a win, draw, or loss respectively [@mcts].

4. **Back-propagation:** With the value obtained from the simulation above, we now update the values of nodes back up the search tree [@mcts]. Now, the node selection will be called again, with slightly or perhaps vastly different values for our selection algorithm [@mcts].

## Convolutional Neural Network Components and Techniques

A Convolutional Neural Network is a common variant of the Artificial Neural Network, often used for object recognition, object detection, and a variety of other problems involving images [@convnet]. It differs from a traditional fully connected ANN in that not all neurons in one layer are connected to all neurons in the next layer [@convnet]. Rather, neurons are only "connected" to neurons that are within close proximity spatially [@convnet]. This significantly reduces the quantity of trainable parameters [@convnet].  While having a lot of parameters is often helpful to avoid underfitting, it is often very difficult to train a network with hundreds of millions of parameters [@convnet]. CNNs also convey spacial information much more efficiently than fully connected layers, as they can be spatially and rotation-ally invariant when necessary [@convnet]. Being able to understand spatial relationships between pieces is likely to be helpful when identifying threats and attacks on the board. In order to propagate information further into the network, filters are "convolved" over the image [@convnet].

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/conv-ty7gT4H2P0yEnpgfddh1hKbPWiRPhx.jpeg">
        <img alt="Convolutional Neural Network" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/conv-ty7gT4H2P0yEnpgfddh1hKbPWiRPhx.jpeg" width="512" height="512" />
    </a>
</p>
Figure 1: Convolution Operation [@convnet]

**Batch Normalization** is a technique that ensures the outputs from previous layers remain relatively close to a standard normal distribution, with a mean of zero and a standard deviation of one [@batchnorm]. This can greatly accelerate the learning process [@batchnorm].
The **Rectified Linear Unit (ReLU)** activation function is a commonly used activation function in Deep Neural Networks [@relu].

$$
  ReLU(x) =
  \begin{cases}
    x & \text{if $x > 0$} \\
    0 & \text{otherwise}
  \end{cases}
$$

Figure 2: ReLU [@relu]

The ReLU activation function is especially useful in Deep Networks as it helps to dodge the vanishing gradient problem as the gradients are relatively stable at different points of the function (0 or 1) [@relu].
**Dropout** is a regularization technique commonly used in Neural Networks [@dropout]. Dropout works by disabling the activation of certain neurons on each training iteration randomly [@dropout]. This ensures knowledge becomes distributed throughout the network and prevents the network from relying on the output of few features [@dropout].
**The Hyperbolic Tangent (tanh)** activation function maps the output of the perceptron to be between -1 and 1 [@tanh]. The equation is:
$$
\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
Figure 3: Hyperbolic Tangent Function [@tanh]

It is relatively linear near 0, and asymptotically approaches $-\infty$ and $\infty$ as you move towards $-1$ and $1$ respectively [@tanh].
**Adam** is an optimizer similar to Gradient Descent, but has additional features that make it more efficient at finding the global minima of the loss function [@adam].

# Related Work

Instead of utilizing a tree search algorithm from the onset, we could begin with a hash table to look up a current board state and determine if we can use book moves to quickly win a game [@stock]. Stockfish employs a similar technique in this sense [@stock]. We ultimately decided not to do this because, although it likely would win more games, it would go against the purpose of using MCTS and a NN to beat another player as weaker players would very easily be thwarted by this method. In addition, it would significantly slow down our engine [@stock].

## AlphaGo Zero

Our biggest inspiration is derived from _AlphaGo Zero_. AlphaGo Zero teaches itself how to play GO with no prior knowledge [@agozero]. It learns the game by combining a neural network with a powerful search algorithm (MCTS) [@agozero]. The NN is updated to learn the eventual winner of games from a state, and is therefore better able to accurately predict moves [@agozero].
In contrast to it's predecessor, _AlphaGo_, AlphaGo Zero only uses the pieces on the board as the input instead of a few human engineered features [@agozero]. Additionally, there is only one NN instead of two [@agozero]. This is accomplished by combining the "policy network" and "value network" into one network to allow for quicker and more effecient training and evaluation [@agozero]. Finally, the NN that is used to predict which player will win is a new addition to AlphaGo Zero [@agozero].
It has been hypothesized that the breakthroughs brought on from AlphaGo Zero can contribute to solving complex problems such as as protein folding, reducing energy consumption, or discovering new materials [@agozero]. Such discoveries would positively impact our society [@agozero].

## Stockfish

The main techniques Stockfish uses to be an effective agent are storing and evaluating data efficiently and proceeding with a recursive search of moves [@stock]. Most of this is accomplished by the ever-evolving alpha-beta pruning algorithm Stockfish employs  [@stock].
Here are a few key things that Stockfish tracks that we do not:

* **Pawn Structure**
    1. "Penalize doubled, backward and blocked pawns" [@stock].
    2. "Encourage pawn advancement where adequately defended" [@stock].
    3. "Encourage control of the center of the board" [@stock].
* **Piece Placement**
    1. "Encourage knights to occupy the center of the board" [@stock].
    2. "Encourage queens and rooks to defend each other and attack" [@stock].
    3. "Encourage 7th rank attacks for rooks" [@stock].
* **Passed Pawns**
    1. "These deserve a special treatment as they are so important" [@stock].
    2. "Check for safety from opposing king and enemy pieces" [@stock].
    3. "Add enormous incentives for passed pawns near promotion" [@stock].
* **King Safety**
    1. "Encourage the king to stay to the corner in the middlegame" [@stock].
    2. "Try to retain an effective pawn shield" [@stock].
    3. "Try to stop enemy pieces from getting near to the king" [@stock].

Despite this, it will soon be evident that we can compete against weaker versions of Stockfish thanks to the generalization prowess of our NN.

# Project Description

The code for the problem at hand, chess, used the implementation of chess from _python-chess_, a widely popularized and open source library written for python.

## Formal Problem Description

### State Space

A valid state includes an $8\times8$ grid. This grid can be occupied by pawns, bishops, knights, rooks, queens, and kings for either player. Valid states are subject to the rules of chess. The state space includes all valid chess positions.
To represent this, we used _Forsyth–Edwards Notation_ (FEN). As an example, here is the starting position in FEN:

$$
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
$$

Figure 4: Forsyth–Edwards Notation (FEN)

The two numbers after the dash are the **halfmove clock** and the **fullmove number** [@fen]. The halfmove clock is the number of halfmoves since the last capture or pawn advance. The fullmove number is the number of full moves that have been played. [@fen]. This representation allowed us to keep track of all pieces on the board in addition to be a hashable version of a board state. We discard the non board state components of the fen as we can track these separately as necessary. [@fen].

### Transition Function

Our transition function takes

1. **A state, $s$,** represented by a FEN string and is defined by an $8\times8$ chess board occupied by up to $32$ pieces.

2. **An action, $a$,** which is the player's piece that is being moved, as represented by a _Universal Chess Interface_ (UCI) string. This string represents the action or move a particular player is performing [@uci]. The string includes the start square and end square for a piece. For example, to move the e pawn two squares forward from the starting position, UCI would encode this as "e2e4" [@uci]. Each action was verified to be a legal action according to the rules of chess applying to the state $s$.
and returns a new chess board state.

### Start State

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/starting_board-mDpgvERxjhFVgzCqDKd7V3ZBCMa3Xe.png">
        <img alt="starting board" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/starting_board-mDpgvERxjhFVgzCqDKd7V3ZBCMa3Xe.png" width="512" height="512" />
    </a>
</p>

Figure 5: The starting board of a chess game

Figure 4 contains the starting FEN for a game of chess. Figure 5 depicts the starting board state as an image.

### Goal Test

* Have we checkmated our opponent? (+1 Reward)
* Have we been Checkmated by our opponent [@stock]? (-1)
* Is either player Stalemated [@stock]? (0) Is there insufficient material [@stock]? (0)
* Has the current board been repeated 3 times [@stock]? (0)
* Have 75 moves occurred since the last pawn capture [@stock]? (0)

Notably, we do not track information regarding threefold repetition within our state. Threefold repetition tracking would have forced us to include every previous board position in the definition of the current state, which would significantly increase the computation required. However, this does not mean we have to disregard this information. Rather, when evaluating a state, we simply compare the fen to our cached previous board positions, and if the position has already occurred, we change the value of that move to be 0 regardless of what our network claims.

## Inputs and Outputs

**Input**: The input to our algorithm was the state (a fen representing the board occupied by a certain arrangement of pieces).
**Output**: The predicted optimal move given the current state of the game.

# Data & Interest

It should be noted we did not use pre-existing datasets for our agent. Our dataset was created manually through self play, via a process that will be discussed in detail in a later section.

## Initial Planned Implementation

Initially, we planned to replicate the AlphaGo Zero paper as closely as possible. Ultimately, we had to modify this implementation to resolve some impracticalities. The corresponding changes will be discussed in a later section. The following information applies to our original design.
AlphaGo Zero has two main components to it:

* The Neural Network, which performs the policy evaluation, as well as predicts the value of a certain state [@stanford].

* The MCTS Component, which performs policy iteration to improve gameplay, and ultimately, the network [@stanford].

## The Neural Network

First, there is a Neural Network $F_{\theta}(s)$, which takes in the current state and outputs the value of that state $v_{\theta}(s)$, as well as a predicted policy, which contains the probability that a certain action should be taken [@stanford].
The network is trained in an iterative manner through self play  [@stanford]. At the end of each game of self play, training examples are provided to the network in the form of
($s_t, \vec{\pi}_t, z_t$) [@stanford]. $\vec{\pi}_t$ is an estimate of what policy would be produced in that state $s_t$, while $z_t$ contains the final outcome of the game (1 if player 1 wins, -1 if player 2 wins, etc.) [@stanford]. Over time, the network improves at predicting which side is going to win from a certain state.
We'd train to minimize the following Loss Function:

$$
l=\sum_{t}(v_{\theta}(s_{t})-z_{t})^2-\vec{\pi}\cdot log(\vec{p}_{\theta}(s_{t}))
$$

Figure 6: Loss Function [@stanford]

## CNN

The initial network was designed to perform for the standard Alpha Go implementation. As such, this network took in a board as it's input, and returned the value of the state as well as the expected policy from that position (the probability that each move should be taken).

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/initial_network-a6f2X7MFL8m9tqr9xl2Ocl9XmgjvMw.png">
        <img alt="intial network" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/initial_network-a6f2X7MFL8m9tqr9xl2Ocl9XmgjvMw.png" width="805" height="474" />
    </a>
</p>

Figure 7: Initial Network Architecture

The chosen network architecture was a _Convolution Neural Network_ (CNN). In order to pass the board in to a Convolutional Neural Network in an efficient manner, inspiration was taken from color images. To pass a color image into a CNN, the image is traditionally split into its color channels (and occasionally an alpha channel for transparency) [@convnet]. Likewise, the chessboard was split into twelve component channels. Each channel contained information regarding the presence of a certain piece of a certain color. Since each player can have pieces of six different types, there were twelve channels total. A single channel would be an eight by eight two-dimensional Tensor with ones in the squares a piece was present in and a zero in the cells a piece was absent.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/piece_breakdown-vb1NIfsCj7sVmJyivD9BPMsCpcvhKW.jpeg">
        <img alt="piece visualization breakdown" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/piece_breakdown-vb1NIfsCj7sVmJyivD9BPMsCpcvhKW.jpeg" width="526" height="711" />
    </a>
</p>

Figure 8: A Visualization of how we represent our boards in the CNN. The board has been split into 12 channels, one for each piece of a certain color. Green represents the presence of a 1, while Black represents the presence of a 0.

For consistency, this was always done from the perspective of white. To play for black, the board was "mirrored" such that the colors were flipped and the pieces were translated so that black's position would be a valid position as white.
The initial network architecture was inspired by the Othello network in the Stanford GitHub [@github]. This network had four convolutional blocks. In each block, the filters from the previous layer were convolved by 512, three by three filters. This convolution was performed with a stride of one.
In the first two blocks padding was used, while in the later blocks padding becomes less important as the features that are inputted are higher level anyway, and thus it is less likely for information to be lost [@convnet].
In each convolution block, batch normalization was used [@batchnorm].
Following our convolutional blocks, we flattened our result into a pair of fully connected layers which fed into our output layer [@convnet].
The Rectified Linear Unit (ReLU) was used as the activation function for all the layers except the output layers [@relu].
In the final layer, the Tanh activation function was utilized for the value output [@tanh].  This is ideal for our board evaluations, as there is a huge difference in being up 3 points of material vs being down 3 points of material. There is not a huge difference between being up 50 points of material and 44 points of material.
Dropout Regularization was applied to the fully connected layers to prevent over fitting [@dropout].
The loss function utilized for the value output was mean squared error. Despite utilizing Tanh as our output layer, we were not performing classification, but rather performing regression and trying to squeeze the results of our regression into an appropriate range. Mean squared error is preferable to mean absolute error as it punishes outliers significantly more [@mse]. Log-Softmax was used as the output layer for the policy to normalize the probabilities [@stanford].
The Adam optimizer was the optimizer of choice [@adam].

## Self Play Phase

We initialize our neural network with semi-random weights resulting in a random policy and network [@stanford].
On each turn, we perform a fixed number of MCTS simulations, let's say 2000 [@stanford]. We sample from our current policy to choose our move [@stanford]. We can produce a training example from this to later determine whether or not that was a good move [@stanford].
Once the game comes to an end, we retrain the network, and we pit the new network against the old one [@stanford]. If the new network beats the old network by a decent margin ($55\%$ of games for example), then that network replaces the old one [@stanford].

## MCTS

In our tree search, we utilize the following:

* $Q(s,a)$, which contains the perceived quality of an action from a state [@stanford].

* $N(s,a)$, which contains the number of times that action has been visited in our search [@stanford].

* $P(s,\cdot)$ which contains the initial estimate of taking the action based on the policy from our CNN [@stanford].

We utilize these to calculate $U(s,a)$, which is an upper bound on the $Q$ value for Action $a$ [@stanford].

$$
U(s,a)=Q(s,a)+c_{puct}\cdot P(s,a) \cdot \frac{\sqrt{\Sigma_b N(s,b)}}{1+N(s,a)}
$$

Figure 9: Upper Confidence Bound on Q-Values [@stanford]

$C_{puct}$ is a hyperparameter that determines how much exploration vs exploitation we perform [@stanford]. We initialize our tree to be empty, with s as the root. We then perform some amount of simulations, lets say 2000.

To Perform a simulation, we do the following:

* "We compute the action that maximizes $U(s,a)$" [@stanford].

* "If we've already seen that node $s'$, we recursively call on $s'$" [@stanford].

* "If it does not already exist, we initialize the state in our tree. We initialize $P(s, \cdot)$ to be our initial estimate of the policy, our predicted value of S to be that given to us by our Neural Network, and we initialize $N(s,a)$ and $Q(s,a)$ to be 0 for all our actions.

  * At this point, we don't perform a rollout but instead predict how good our state is via our neural network. We propogate this information back up the tree to adjust our predictions for how good the value of the above state is. If we happen to encounter a terminal state, we simply propagate that value up instead" [@stanford].

* "Once we've performed our iterations, rather than using our $Q$ values, our $N$ Values represent a good approximation for our policy (after all, if those moves were not promising, we wouldn't be visiting them so often). Our updated policy is simply the normalized counts (divide the number that this action was taken by the number of times all the actions were taken)" [@stanford].

* "During our self play phase, we pick a move by sampling from this improved policy" [@stanford].

## Impracticality

After experimentation, it became clear that the AlphaGo approach would not be particularly practical given the computational resources available. Once this became clear, we elected to update our network to no longer attempt to predict the policy from a position. Consequently we eliminated the components designed to infer the policy. Once we had a proof of concept, we iteratively made improvements to our network. Detailed results for each of these improvements can be found in the Experimentation section.

## Training Data Generation

Training Data was initially produced in a manner similar to the Alpha-Go paper, with our MCTS implementation playing itself and recording whether a certain board position that was reached resulted in a win, loss, or draw [@stanford].
Unfortunately, this proved to be prohibitively slow, with only one game being played every minute. In addition, almost every game resulted in a drawn position, so very little information was being conveyed.

In Catfish ML V1, we instead decided to base our board evaluations off of Stockfish evaluations, and have Stockfish play itself 5000 times. Each training example contained the board after each move stored as an array (as defined by our split-channel notation) as well as a number depicting Stockfish's evaluation of the current state. In other words, our initial networks goal was to pick the best move in a given position. Our new network's goal was to predict how Stockfish would evaluate a board position. We're effectively trying to extract information from Stockfish and put it in our own engine.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/data_generation-LDoFnSP1h4d33ftqxg8akpzDAs2zv0.jpeg">
        <img alt="sample training example" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/data_generation-LDoFnSP1h4d33ftqxg8akpzDAs2zv0.jpeg" width="597" height="419" />
    </a>
</p>
Figure 10: A sample training example from our generated dataset.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td1-957zLy6E13nMbzN0Z4slPsZCVD5spV.png">
        <img alt="training data v1" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td1-957zLy6E13nMbzN0Z4slPsZCVD5spV.png" width="640" height="480" />
    </a>
</p>
Figure 11: The training data distribution of value labels used for Catfish V1

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd1-ap8IsLdUIdueAxT24mQ7iHSZQpBR47.png">
        <img alt="validation data v1" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd1-ap8IsLdUIdueAxT24mQ7iHSZQpBR47.png" width="640" height="480" />
    </a>
</p>
Figure 12: The validation data distribution of value labels used for Catfish V1

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td2-Jc8W2W55fVyQNJn9mWQMHifcFK9ncC.png">
        <img alt="training data v2" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td2-Jc8W2W55fVyQNJn9mWQMHifcFK9ncC.png" width="640" height="480" />
    </a>
</p>
Figure 13: The training data distribution of value labels used for Catfish V2

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd2-eiTFwnnXa4Wzw4pK7jjrOuKzhR1xCQ.png">
        <img alt="validation data v2" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd2-eiTFwnnXa4Wzw4pK7jjrOuKzhR1xCQ.png" width="640" height="480" />
    </a>
</p>
Figure 14: The validation data distribution of value labels for Catfish V2

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td3-exNBJTAukF3BPvFGxhni62Nvu0m2Ai.png">
        <img alt="training data v3" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/td3-exNBJTAukF3BPvFGxhni62Nvu0m2Ai.png" width="640" height="480" />
    </a>
</p>
Figure 15: The training data distribution of value labels for Catfish V3

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd3-ph5LsUNJwS9knPy8paKPOoKd3xnPAV.png">
        <img alt="validation data v3" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/vd3-ph5LsUNJwS9knPy8paKPOoKd3xnPAV.png" width="640" height="480" />
    </a>
</p>
Figure 16: The validation data distribution of value labels for Catfish V3

This had multiple advantages over our previous approaches. First, significantly more games ended in wins and losses as opposed to only draws, meaning that much more information was conveyed. Secondly, games could be played much more quickly, with Stockfish choosing a move in a fraction of a second. Ultimately, this allowed us to collect much more viable training data.

## Larger Network

After achieving a proof of concept, we decided to increase the complexity of our architecture. Components of the original network were maintained, with adjustments made to incorporate Resnet50. Resnet50 is a Deep Convolutional Neural Network architecture commonly used on object detection and localization problems [@resnet]. The most iconic feature of Resnet is that it contains skip connections [@resnet]. Skip connections make it possible to train significantly deeper networks because it diminishes the extent of the vanishing gradient problem. [@resnet].

```python
class OthelloNNet(nn.Module):
    def __init__ (self, args):
        self action_size = (42722)
        self.args = args
        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(12, args["num_channels"], (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(args ["num_channels"], args["num_channels"], (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(args ["num_channels"], args["num_channels"], (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(args["num_channels"], 3, (3, 3), stride=(1, 1))

        self.resnet = resnet50(progress=False)

        self.bn1 = nn.BatchNorm2d(args["num_channels"])
        self.bn2 = nn.BatchNorm2d(args["num_channels"])
        self.bn3 = nn.BatchNorm2d(args["num_channels"])
        self.bn4 = nn.BatchNorm2d(3)

        self.fc2 = nn.Linear(1000, 512)
        self.fc_bn1 = nn.BatchNorm1d(1000)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear (512, 1)

    def forward (self, s):
        s = s.View(-1, 12, 8, 8)
        s = Frelu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = F.relu(self.resnet(s))

        s = F.dropout(F.relu(self.fc_bn1(s)), p=self.args['dropout'], training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args[ 'dropout'], training=self.training)
        
        v = self.fc4(s)

        return torch.tanh(v)
```

Figure 17: Updated Network Architecture

To accommodate ResNet, we we updated the final convolutional block to only output 3 filters [@resnet]. This is consistent with the 3 Color channels that are typically used as inputs to Resnet50 [@resnet]. We also updated our first Linear layer to have the appropriate number of input features (1000).

# Experiments

## Defintions

A **pseudo-win** is a draw that ends with us up in material. Similarly, a **pseudo-loss** is a draw that ends with us down in material and a **pseudo-draw** is a draw with equal material for both players.

## Exploration

As an exploration to test our MCTS implementation, we bypassed our planned Neural Network roll-out and instead did a roll-out with _StockFish 13_ whenever a state was encountered for the first time. The value of this state was determined by having the engine evaluate the current board state and returning the relative score. To update the policy, we had the engine choose the best action from the possible legal actions. The best action was given a value of $1$, while all other moves were $0$.

We then ran the algorithm against itself until the game was over. This allowed the search tree to populate with examples from these games. While the game wasn't over, we ran the _MCTS_ algorithm $20$ times per turn. The selected move was chosen as a weighted average between all legal moves [@github]. This weighting was determined by the following:

$$
[w=[\frac{N(s,a_1) ^{temp^{-1}}}{y}, \frac{N(s,a_2) ^{temp^{-1}}}{y}, ... , \frac{N(s,a_n) ^{temp^{-1}}}{y}]]
$$

Figure 18: Move Weighting [@github]

Where $n$ is the number of legal actions in a board state, $temp$ is $1$ whenever we have completed less than $15$ moves in the game and $0$ otherwise, and where $y$ is the sum of all $n$ values from state $s$ and action $a$ defined as follows [@github]:

$$
[y=\sum_{a=1}^{a=n}N(s,a)]
$$

Figure 19: $y$ as defined for _Fig. 4_ [@github]

Then we pitted our agent against a random agent. The random agent simply chose randomly from all legal moves, and our agent chose based on _N_ values as defined above.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/MCTS_Stockfish_vs_Random-sBoukRzQBsqzlcgBbafl8lsgR2sQvS.png">
        <img alt="mcts vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/MCTS_Stockfish_vs_Random-sBoukRzQBsqzlcgBbafl8lsgR2sQvS.png" width="600" height="371" />
    </a>
</p>

Figure 20: MCTS vs a Random Agent

After $200$ games, we won $190$ times against the random agent, and ended in a pseudo-win with the remaining $10$ games. It should be noted a random mover has an Elo of approximately $245$ [@rndelo]. Based on these preliminary results, it appeared our _MCTS_ algorithm was working as intended.
We then underwent several rounds of experimentation. Each change resulted in a significant improvement over the prior iteration and incorporated new understanding.

## Initial Performance (Catfish V0)

Our first approach was to attempt to replicate the Alpha-Go paper as is. The performance of this was very disappointing and for good reason. We lacked the computational resources to play a significant amount of games, and most of our games ended in draws. Consequently, we learned almost nothing per iteration, and had to adjust our approach.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V0_vs_Random-idgmRYNswYmZcR3nOY58FOaAMFOwVY.png">
        <img alt="catfish v0 vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V0_vs_Random-idgmRYNswYmZcR3nOY58FOaAMFOwVY.png" width="600" height="371" />
    </a>
</p>

Figure 21: Catfish V0 vs a Random Agent

## First Stockfish Self Play Results (Catfish V1)

Next, we shifted our approach to instead train off Stockfish 13 self play, rather than our own games. For each board position either Stockfish encountered, we recorded the board and Stockfish's evaluation of who was winning. By doing this, we got significantly better results vs random. Most notably, we did not lose a single game.
However, against any agent that was even remotely adversarial, Catfish V1 was demolished. Even Stockfish level 1 (Elo 250) [@chesscom] played us to a pseudo-draw. Clearly there were improvements to be made.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V1_vs_Random-YAXz5viBoBIlzp7xo9z7uZxkCZJhTP.png">
        <img alt="catfish v1 vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V1_vs_Random-YAXz5viBoBIlzp7xo9z7uZxkCZJhTP.png" width="600" height="371" />
    </a>
</p>

Figure 22: Catfish V1 vs a Random Agent

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv1-3jOjtxHv1HO3InHeQN7lAub8bpIonp.png">
        <img alt="catfish v1 training validation loss" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv1-3jOjtxHv1HO3InHeQN7lAub8bpIonp.png" width="640" height="480" />
    </a>
</p>

Figure 23: The training and validation loss while training Catfish V1

## Checks and Balances

Given the significant quantity of pseudo-wins, we sought out to convert these into actual victories. First, we set it up so if there was a move that would achieve checkmate, we'd automatically take the move, regardless of evaluation. Next, we kept track of previous board positions in a hash table. If the move we selected would lead to an existing board position, we updated the evaluation of that position to 0, regardless of what the network predicted. Likewise, if a board position would lead to stalemate, we updated the evaluation of that position to 0, regardless of our networks prediction.

## Only using "Black To Move" Boards (Catfish V2)

Finally, we realized the main reason for our poor performance against Stockfish. Our training examples were containing boards from both white and black's perspective. However, during gameplay, we should only be evaluating boards as if it's blacks turn to move (since no matter what move we make, it will always be blacks turn next). As chess is a turn based game, the next player to move from a position is crucial information for evaluating a position.

To resolve this issue, we retrained our model, only saving training examples where it was blacks turn to move next. This resulted in significant improvements. Catfish ML V2 was able to play Stockfish Levels 1-5 (ELO 250-850) [@chesscom] to Pseudo Wins, ending these games with significant material advantages and often mate in several moves. However, Catfish ML V2 was still struggling to achieve checkmate against adversarial opponents.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V2_vs_Random-IquqybD1eRPG0VlNxX7G2HUSTEwQVE.png">
        <img alt="catfish v2 vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V2_vs_Random-IquqybD1eRPG0VlNxX7G2HUSTEwQVE.png" width="600" height="371" />
    </a>
</p>
Figure 24: Catfish V2 vs a Random Agent

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv2-Pq9ss2lla6HCTIWfAOtkV7HUT02FzF.png">
        <img alt="catfish v2 training validation loss" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv2-Pq9ss2lla6HCTIWfAOtkV7HUT02FzF.png" width="640" height="480" />
    </a>
</p>

Figure 25: The training and validation loss while training Catfish V2

## Updated Network that seeks out Checkmate (Catfish V3)

At this point, we were considering what other improvements could be made. The remaining amount of pseudo-wins were making us very uncomfortable, as given the amount of material we were up, they should have been very easy to convert into wins. There were a few avenues that we could have utilized to resolve this. Our first idea was to incorporate an exhaustive tree search on the engine that would only run during endgames, but this did not seem in the spirit of the rest of our implementation, nor did it seem particularly efficient. This approach would also miss early mating opportunities in games.

It occurred to us that perhaps the solution was to return to our network and attempt to teach our model some mating patterns. We went back to our training data generation so our examples would value positions closer to mate as stronger than positions that were further from mate. We updated the evaluation for a board near mate to be tanh(100,000 / mate-in-x-moves).
This change was extremely successful and eliminated virtually all games vs random that previously went to 75 moves. Our results for this batch of 10000 games were very satisfactory, as we ultimately achieved a 99.85\% win-rate in the 10000 games we played vs the bot. Additionally, we now were able to checkmate Stockfish 5, where previously we struggled to make progress once we had a significant material advantage.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V3_vs_Random-w6TZMee8kROVA8MKnm3mC3huaHkkzn.png">
        <img alt="catfish v3 vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/Catfish_V3_vs_Random-w6TZMee8kROVA8MKnm3mC3huaHkkzn.png" width="600" height="371" />
    </a>
</p>

Figure 26: Catfish V3 vs a Random Agent

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv3-weg3ON5U07R8uz1ne8iuZSlRLT2d7C.png">
        <img alt="catfish v3 training validation loss" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/tv3-weg3ON5U07R8uz1ne8iuZSlRLT2d7C.png" width="640" height="480" />
    </a>
</p>

Figure 27: The training and validation loss while training Catfish V3

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/catfish2_beating_sf_lvl_5-hBXSfD8ltRdtI1R6hUY9s5AeKbyddY.jpeg">
        <img alt="catfish v3 wins" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/catfish2_beating_sf_lvl_5-hBXSfD8ltRdtI1R6hUY9s5AeKbyddY.jpeg" width="512" height="512" />
    </a>
</p>

Figure 28: Screen Capture of Catfish V3 (white) vs Stockfish level 5 (black). The Black King has been checkmated, so white (Catfish V3) has won the game.

## MCTS and the Neural Net

Finally we put our neural network back into our Monte Carlo Tree Search implementation, replacing Stockfish. Right away, we noticed that games were taking drastically longer to play compared to Catfish V3 on its own, as rather than simply taking the best move from the NN, the MCTS algorithm involves us recursively predicting all legal moves down to a certain depth.

<p align="center">
    <a href="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/MCTS_Catfish_V3_vs_Random-qwJGo8uZ7KriuDyE33Z60sOb12IP1y.png">
        <img alt="mcts_catfish v3 vs random agent" src="https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/blogs/monte-carlo/MCTS_Catfish_V3_vs_Random-qwJGo8uZ7KriuDyE33Z60sOb12IP1y.png" width="600" height="371" />
    </a>
</p>

Figure 29: MCTS & Catfish V3 vs a Random Agent

As previously mentioned, the difference between a two-ply search and a three-ply search is significant (900 vs 27,000 possibilities) [@chess]. Because of this, there simply was not enough time to run as many games of this agent against a random agent like we did for Catfish V3.

$$
\begin{array}{l|cccccc}
     & Wins & Pseudo-Wins & Pseudo-Draws & Pseudo-Losses & Draws & Losses \\\hline
    \text{MCTS} &190 &10 &0 &0 &0 &0\\
    \text{Catfish V0} &0 &0 &0 &0 &36 &4\\
    \text{Catfish V1} &41 &159 &0 &0 &0 &0\\
    \text{Catfish V2} &184 &16 &0 &0 &0 &0\\
    \text{Catfish V3} &9986 &10 &0 &0 &0 &4\\
    \text{MCTS \& Catfish V3} &192 &7 &0 &0 &0 &1\\
\end{array}
$$

Figure 30: Our agents vs a random agent

# Conclusion

There are several avenues available to build on this research. Now that there is a proof of concept that a neural engine for chess can be trained for chess on reasonably accessible hardware, it may become worthwhile to attempt to shift back towards the original Alpha-Go Paper. In other words, we could realistically store our self play games as well as our chosen policy and the result of the games, and then try to train with policy directly.
In addition, it may make sense to attempt to support our network with a more traditional heuristic model. Against Stockfish level 6 (Elo 1000) [@chesscom], we were winning for a period of time before blundering a Queen. It could be reasoned that conducting a local tree search on the move we're considering to ensure it's not actually a really poor move could significantly increase our ELO.

Checkmate aversion was not taken into account during both our training and in our move decision function. As a result, MCTS & Catfish actually occasionally lost a game to a random agent. There are two clear ways to remedy this. One is to modify the evaluation of the boards in our training examples so our NN can better foresee being checkmated. The other involves ignoring the valuation returned by our NN when a move would lead to our opponent having mate-in-one and setting the move value to $-1$. One or both of these could virtually eliminate the chances of defeat against a random opponent.

We learned a lot from completing this project. First, we learned that it is important to confirm that the computational resources available to you are reasonable in the context of the paper you are trying to replicate the results of. If we had realized up front that Alpha-Zero could self-play tens of thousands of times faster than us [@agozero], it is unlikely we would've undergone this project.

In addition, we learned the efficiency of iterating through model architectures without simply jumping to the most powerful one we could find. We were able to make incremental yet meaningful improvements in each generation of Catfish, greatly improving over the performance of the previous generation.
At the start of this project, we set out to play a game of chess intelligently. Based on our experimental results, it is apparent we were able to accomplish this with our exploration into convolutional neural networks and Monte Carlo Tree Search.

<!-- 

Unfortunately I cannot find these files on S3 or locally. Leaving commented out until they can be found.

# Checkpoints & Data

We stored the data used for training our models and their corresponding checkpoints on _Amazon Web Services S3_. These can be retrieved at the links below.

* Catfish V1 Checkpoint:     <https://zachjarchessstar.s3.amazonaws.com/checkpoint3.pth.tar>
* Catfish V2 Checkpoint: <https://zachjarchessstar.s3.amazonaws.com/checkpoint5original.pth.tar>
* Catfish V3 Checkpoint: <https://zachjarchessstar.s3.amazonaws.com/checkpoint5.pth.tar>
* Catfish V1 Training Data: <https://zachjarchessstar.s3.amazonaws.com/outfile3>
* Catfish V1 Validation Data: <https://zachjarchessstar.s3.amazonaws.com/outfile4>
* Catfish V2 Training Data: <https://zachjarchessstar.s3.amazonaws.com/outfile5>
* Catfish V2 Validation Data: <https://zachjarchessstar.s3.amazonaws.com/outfile6>
* Catfish V3 Training Data: <https://zachjarchessstar.s3.amazonaws.com/outfile7>
* Catfish V3 Validation Data: <https://zachjarchessstar.s3.amazonaws.com/outfile8> -->

**References:**

[^ref]
