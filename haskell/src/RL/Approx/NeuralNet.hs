{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Use Neural Network to approximate the value function



module RL.Approx.NeuralNet where

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , (<.>)
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

scoreLayer :: Vector R -> Layer -> Vector R
scoreLayer input Layer { weights, activation } = cmap activation ( tr' weights #> input )

-- Need to look at the foldr and needing to flip the order
scoreNeuralNetwork :: Vector R -> [Layer] -> Vector R
scoreNeuralNetwork input lls = foldr (.) id (map scoreLayer lls) input 
