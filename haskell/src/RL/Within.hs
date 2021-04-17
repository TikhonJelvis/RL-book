{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}
module RL.Within where

import           Numeric.LinearAlgebra                    ( Matrix
                                                          , R
                                                          , Vector
                                                          )

import           RL.Matrix                                ( allWithin )

class Within a where
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
         -> a
         -> a
         -> Bool

instance Within R where
  within ϵ a b = abs (a - b) <= ϵ

instance Within (Vector R) where
  within = allWithin

instance Within (Matrix R) where
  within = allWithin
