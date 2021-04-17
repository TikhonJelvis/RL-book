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
                                                          , scalar
                                                          , scale
                                                          , tr'
                                                          , (|>)
                                                          )

import           Text.Printf                              ( printf )


import           RL.Matrix                                ( (<$$>) )
import           RL.Vector                                ( Affine(..) )

import           RL.Approx.Approx                         ( Approx(..) )
import           RL.Approx.Weights                        ( Weights
                                                          , values
                                                          )
import qualified RL.Approx.Weights                       as Weights

-- | An approximation that uses a linear combination of /features/
-- ('ϕ') and an extra regularization term ('λ') to model a function.
data Linear a = Linear
  { ϕ :: !(a -> Vector R)
    -- ^ Get all the features for an input @a@.
  , w :: !Weights
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
create λ fs = Linear { λ, ϕ = features fs, w = Weights.init (length fs) 0 }

lossGradient :: Linear a -> V.Vector a -> Vector R -> Vector R
lossGradient Linear { ϕ, w, λ } x y = (tr' ϕₓ #> (y' - y)) / n + reg
 where
  ϕₓ  = ϕ <$$> x
  y'  = ϕₓ #> values w
  n   = scalar (fromIntegral $ length x)
  reg = scale λ $ values w

instance Affine (Linear a) where
  type Diff (Linear a) = Vector R

  l₁ .-. l₂ = w l₁ .-. w l₂

  l .+ v = l { w = w l .+ v }


instance Approx Linear a where
  eval Linear { ϕ, w } a = ϕ a <.> values w

  update linear@Linear { w } x y =
    linear { w = Weights.update w $ lossGradient linear x y }

  within ϵ l₁ l₂ = Weights.within ϵ (w l₁) (w l₂)

