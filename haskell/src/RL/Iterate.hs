module RL.Iterate where

import qualified Data.List                               as List

-- | Return the first element of the list once the given condition is
-- satisfied on a consecutive pair of elements.
--
-- Returns 'Nothing' if the list ends before converging. Loops forever
-- on infinite lists that don't converge.
--
-- @
-- λ> converge (\ a b -> a / b > 0.8) [0..]
-- Just 5.0
-- λ> converge (/=) "aabc"
-- Just 'a'
-- λ> converge (==) [1.0]
-- Nothing
-- λ> converge (==) [1.0..]
-- <loop forever>
-- @
converge :: (a -> a -> Bool) -> [a] -> Maybe a
converge condition xs = do
  let condition' (a, b) = condition a b
  (a, _) <- List.find condition' $ zip xs (drop 1 xs)
  pure a

