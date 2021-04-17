{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ParallelListComp #-}
-- | 'Approx' instances for __dynamic programming__: using a table to
-- represent a function with a finite domain /exactly/.
--
-- While the 'Approx' class is primarily meant for function
-- approximations, it can also be used for exact dynamic
-- programming. This lets us have a single implementation of
-- algorithms like Bellman iteration that can be run in both "exact"
-- and "approximate" modes.
module RL.Approx.Dynamic where

import qualified Data.Foldable                           as Foldable
import           Data.HashMap.Strict                      ( (!)
                                                          , HashMap
                                                          )
import qualified Data.HashMap.Strict                     as HashMap
import           Data.Hashable                            ( Hashable )
import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( R )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Approx.Approx                         ( Approx(..) )

-- | An 'Approx' that models a function as a map.
data Dynamic a = Dynamic
  { mapping :: HashMap a R
  }
  deriving (Show, Eq)

instance (Eq a, Hashable a) => Approx Dynamic a where
  eval Dynamic { mapping } x = mapping ! x

  update Dynamic { mapping } xs ys = Dynamic
    { mapping = HashMap.fromList pairs <> mapping
    }
    where pairs = [ (x, y) | x <- V.toList xs | y <- Matrix.toList ys ]

  within ϵ (Dynamic d₁) (Dynamic d₂) =
    (HashMap.keys d₁ == HashMap.keys d₂)
      && all within_ϵ (Foldable.toList d₁ `zip` Foldable.toList d₂)
    where within_ϵ (a, b) = abs (a - b) <= ϵ

-- | Create a 'Dynamic' function approximation for the given set of
-- inputs, all initialized to 0.
create :: (Eq a, Hashable a) => [a] -> Dynamic a
create xs = Dynamic { mapping = HashMap.fromList [ (x, 0) | x <- xs ] }
