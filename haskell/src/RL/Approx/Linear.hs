{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Linear approximations—approximate functions as linear
-- combinations of features.
module RL.Approx.Linear where

import qualified Control.Monad                          as Monad
import qualified Control.Monad.Bayes.Class              as Probability
import qualified Control.Monad.Bayes.Sampler            as Probability

import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , (<.>)
                                                          , Matrix
                                                          , R
                                                          , Vector
                                                          , inv
                                                          , scalar
                                                          , size
                                                          , tr'
                                                          , (|>)
                                                          )

import qualified Numeric.LinearAlgebra                   as Matrix

import           Text.Printf                              ( printf )


import           RL.Matrix                                ( (<$$>) )
import qualified RL.Matrix                               as Matrix
import           RL.Vector                                ( Affine(..)
                                                          , VectorSpace(..)
                                                          )
import           RL.Within                                ( Within(..) )


import           RL.Approx                                ( Approx(..), Batch(..) )
import           RL.Approx.Weights                        ( Weights
                                                          , values
                                                          )
import qualified RL.Approx.Weights                       as Weights

-- | An approximation that uses a linear combination of /features/
-- ('ϕ') and an extra regularization term ('λ') to model a function.
data Linear a = Linear
  { ϕ :: !(a -> Vector R)
    -- ^ Get all the features for an input @a@.
  , w :: !(Vector R)
    -- ^ The weights of all the features ϕ. Should have the same
    -- dimension as ϕ returns.
  , λ :: !R
    -- ^ The regularization coefficient.
  }

-- | For debugging and GHCi.
instance Show a => Show (Linear a) where
  show Linear { w, λ } =
    printf "Linear { ϕ = (λ x → …), w = %s, λ = %f }" (show w) λ

-- | Convert a list of feature functions into a single function
-- returning a vector.
-- 
-- Basic idea: @[f₁, f₂, f₃]@ becomes @f(x) = <f₁ x, f₂ x, f₃ x>@.
features :: [a -> R] -> (a -> Vector R)
features fs a = length fs |> [ f a | f <- fs ]

-- | Convert a list of feature functions into a 'Linear'
-- approximation.
create :: R
       -- ^ Regularization coefficient (λ).
       -> [a -> R]
       -- ^ A set of feature functions (ϕ).
       -> Linear a
create λ fs = Linear { λ, ϕ = features fs, w = length fs |> [0, 0 ..] }

lossGradient :: Linear a -> V.Vector a -> Vector R -> Vector R
lossGradient Linear { ϕ, w, λ } x y = (tr' ϕₓ #> (y' - y)) / n + (λ *: w)
 where
  ϕₓ = ϕ <$$> x
  y' = ϕₓ #> w
  n  = scalar (fromIntegral $ length x)

instance Affine (Linear a) where
  type Diff (Linear a) = Vector R

  l₁ .-. l₂ = w l₁ - w l₂

  l .+ v = l { w = w l + v }


instance Approx Linear a where
  eval Linear { ϕ, w } a = ϕ a <.> w

  direction Linear { ϕ, λ } Batch { xs, ys } = inv left #> right
   where
    left  = (tr' ϕₓ <> ϕₓ) + scalar (n * λ)
    right = tr' ϕₓ #> ys

    ϕₓ    = ϕ <$$> xs
    n     = fromIntegral (snd $ size ϕₓ)


instance Within (Linear a) where
  within ϵ l₁ l₂ = Matrix.allWithin ϵ (w l₁) (w l₂)



-----------------------------------------------------------------------------
-- Test Cases
-----------------------------------------------------------------------------

-- Y = Β X + ε
-- Β = [ 1, 3, 5]
-- X = [ [2, 4, 6], [2, 8, 7], [1, 9, 5], [2, 2, 6], [3, 5, 8]]
-- eps = N(0, σ²) where σ = 0.5

b :: Vector Double
b = Matrix.fromList [ 1, 3, 5]

x :: [Vector Double]
x =  fmap Matrix.fromList [[2, 4, 6], [2, 8, 7], [1, 9, 5], [2, 2, 6], [3, 5, 8]]

eps :: IO [Double]
-- fmap for IO is done implicitly! since it can't be passed as an argument
eps = (Monad.replicateM 5) (Probability.sampleIO (Probability.normal 0 0.05) )

y :: IO (Vector Double)

y = do eps' <- eps
       let go xvec epsilon = b <.> xvec + epsilon 
       pure (Matrix.fromList (zipWith go x  eps')) 

-- Now let's create feature functions to setup the linear
linTest :: Linear (Vector Double)
linTest = create 0 [ \x -> Matrix.atIndex x 0,
                     \x -> Matrix.atIndex x 1,
                     \x -> Matrix.atIndex x 2
                   ]       

-- X is a list of Vector Double, i.e. [Vector Double]
-- V.fromList turns it into the vector, but we need asRow to make a list for each row
testDirection = do y' <- y
                   pure (direction linTest Batch {xs = V.fromList x, ys = y'})
