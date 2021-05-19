{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE NamedFieldPuns #-}

-- | A class for __function approximations__: updatable approximations
-- for functions of the type @a → ℝ@ for any @a@.
module RL.Approx.Approx where

import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( R
                                                          , Vector
                                                          )


import qualified RL.Matrix                               as Matrix
import           RL.Vector                                ( Affine(..)
                                                          , VectorSpace(..)
                                                          )

-- | Updatable approximations for functions of the type @a → ℝ@.
class Affine (f a) => Approx f a where
  -- | Evaluate the function approximation as a function.
  --
  -- Another perspective: interpret values of @f a@ as functions
  -- @a → ℝ@.
  eval :: f a -> (a -> R)

  -- | Find an approximation that's reasonably close to the given set
  -- of x, y pairs.
  fit :: f a
      -- ^ Starting configuration, if needed. Should probably ignore
      -- any "updatable" elements like weights... etc.
      -> V.Vector a
      -- ^ x values—should map 1:1 to y values
      -> Vector R
    -- ^ y values—should map 1:1 to x values
      -> f a

  -- | Evaluate a whole bunch of inputs and produce a vector of the
  -- results.
  eval' :: f a -> (V.Vector a -> Vector R)
  eval' f xs = Matrix.storable (eval f <$> xs)

  -- | Improve the function approximation given a list of
  -- observations. The two vectors have to be the same length and
  -- match up one-to-one.
  update :: Scalar (Diff (f a))
         -- ^ α: Learning rate or how much to weigh new
         -- information. @α = 0@ means an update does nothing; @α = 1@
         -- means ignore previously learned values and only fit to new
         -- observations.
         -> f a
         -- ^ Function approximation to start from.
         -> V.Vector a
         -- ^ x values to update
         -> Vector R
         -- ^ y values for each x value to update
         -> f a
  update α f x y = f .+ (α *: (fit f x y .-. f))
