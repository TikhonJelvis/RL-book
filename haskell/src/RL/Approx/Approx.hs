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
import           RL.Vector                                ( Affine )

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
  eval' f xs = Matrix.storable $ eval f <$> xs

  -- | Improve the function approximation given a list of
  -- observations. The two vectors have to be the same length and
  -- match up one-to-one.
  update :: f a
         -- ^ Function approximation to start from.
         -> V.Vector a
         -- ^ x values to update
         -> Vector R
         -- ^ y values for each x value to update
         -> f a

  -- | Are the two function approximations within ϵ of each other?
  --
  -- 'within' can consider implemenation details of the
  -- approximation—it is not necessary that the /functions being
  -- approximated/ are also within ϵ of each other.
  --
  -- This is useful for deciding when to stop iterative algorithms,
  -- but I'm not sure if this really makes sense /for this class/—this
  -- could be moved to its own class or removed altogether in the
  -- future.
  within :: R
         -- ^ ϵ: bound to compare approximations
         -> f a
         -> f a
         -> Bool
