{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedLists #-}

-- | Use Neural Network to approximate the value function



module RL.Approx.NeuralNet where

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , (<.>)
                                                          , (><)
                                                          , Matrix
                                                          , R
                                                          , Vector
                                                          , cmap
                                                          , inv
                                                          , scalar
                                                          , size
                                                          , tr'
                                                          , (|>)
                                                          )



data NeuralNet n = NeuralNet
  { ϕ :: !(n -> Vector R)
    -- ^ Get the inputs to score using a neural network in the required format
  , layers :: [ Layer ]
    -- ^ Describes the underlying neural net for scoring the input
  }

data Layer = Layer
  {
    weights :: Matrix R,
    activation :: (R -> R) 
  }

scoreLayer :: Layer -> Vector R -> Vector R
scoreLayer Layer { weights, activation } input = cmap activation ( tr' weights #> input )

-- Need to look at the foldr and needing to flip the order
scoreNeuralNetwork :: Vector R -> [Layer] -> Vector R
-- scoreNeuralNetwork input lls = foldr (.) id (map (flip scoreLayer lls) input
-- we need to flip, since the natural composition order is R to L, but the
-- composition we want for matrix-multiplication, it needs to be L to R
scoreNeuralNetwork input lls = foldr (flip (.)) id (map scoreLayer lls) input 


-- Create a simple test
-- input x = [1, 1, 1]
-- NeuralNet n =

sampleInput :: Vector R
sampleInput = [1, 1, 1]

-- llayers = [ Matrix [ [] [] [] ] }

-- Create an identity matrix, single layer, reproduces the input
layer :: Layer
layer = Layer{ weights = (3><3) [1, 0, 0,
                                  0, 1, 0,
                                  0, 0, 1
                                ],
                  activation = max 0 
                }

llayers :: [Layer]
llayers = [ layer 
          ]

score1 = scoreNeuralNetwork sampleInput llayers


layers2 = [layer, layer]
score2 = scoreNeuralNetwork sampleInput layers2

-- Create an identity matrix, single layer, add a activation function

layer2a :: Layer
layer2a = Layer{ weights = (3><3) [1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1
                                    ],
                  activation = (2 *)
                }
layers2a = [layer2a, layer2a]
score2a = scoreNeuralNetwork sampleInput layers2a

-- Create another layer, which can be strung with the first layer, but without having same shape

layer3x2 :: Layer

layer3x2 = Layer{ weights = (3><2) [1, 0,
                                    0, 1,
                                    1, 1
                                   ],
                  activation = id
                }

layers3x3a3x2 :: [Layer]
layers3x3a3x2 = [layer, layer3x2]

score3x2a = scoreNeuralNetwork sampleInput layers3x3a3x2
