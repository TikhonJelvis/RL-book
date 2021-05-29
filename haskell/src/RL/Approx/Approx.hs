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

-- | A batch of x, y observations.
data Batch a = Batch
  { xs :: !(V.Vector a)
    -- ^ X values, same length as ys
  , ys :: !(Vector R)
    -- ^ Corresponding Y values, same length as xs
  }


-- | Updatable approximations for functions of the type @a → ℝ@.
class Affine (f a) => Approx f a where
  -- | Evaluate the function approximation as a function.
  --
  -- Another perspective: interpret values of @f a@ as functions
  -- @a → ℝ@.
  eval :: f a -> (a -> R)

  -- | Evaluate a whole bunch of inputs and produce a vector of the
  -- results.
  eval' :: f a -> (V.Vector a -> Vector R)
  eval' f xs = Matrix.storable (eval f <$> xs)

  -- | Given an existing approximation and a batch of additional
  -- points produces a @Diff (f a)@ that would move the existing
  -- observation to better represent the new observations.
  direction :: f a
            -- ^ Existing function approximation.
            -> Batch a
            -- ^ A batch of new observations.
            -> Diff (f a)

  -- | Improve the function approximation given a batch of additional
  -- x, y points.
  update :: Scalar (Diff (f a))
         -- ^ α: Learning rate or how much to weigh new
         -- information. @α = 0@ means an update does nothing; @α = 1@
         -- means ignore previously learned values and only fit to new
         -- observations.
         -> f a
         -- ^ Function approximation to start from.
         -> Batch a
         -- ^ Batch of new points to update the approximation.
         -> f a
  update α f new = f .+ (α *: direction f new)
