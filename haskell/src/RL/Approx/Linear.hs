{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
-- | Linear approximations—approximate functions as linear
-- combinations of features.
module RL.Approx.Linear where

import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( (#>)
                                                          , (<.>)
                                                          , R
                                                          , Vector
                                                          , inv
                                                          , scalar
                                                          , size
                                                          , tr'
                                                          , (|>)
                                                          )

import           Text.Printf                              ( printf )


import           RL.Matrix                                ( (<$$>) )
import qualified RL.Matrix                               as Matrix
import           RL.Vector                                ( Affine(..)
                                                          , VectorSpace(..)
                                                          )
import           RL.Within                                ( Within(..) )


import           RL.Approx                                ( Approx(..) )
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

  fit Linear { ϕ, λ } x y = Linear { ϕ, λ, w = inv left #> right }
   where
    left  = (tr' ϕₓ <> ϕₓ) + scalar (n * λ)
    right = tr' ϕₓ #> y

    ϕₓ    = ϕ <$$> x
    n     = fromIntegral (snd $ size ϕₓ)


instance Within (Linear a) where
  within ϵ l₁ l₂ = Matrix.allWithin ϵ (w l₁) (w l₂)

