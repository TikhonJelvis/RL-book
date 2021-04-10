-- | Matrix operations useful for RL operations.
module RL.Matrix where

import           Numeric.LinearAlgebra                    ( Matrix
                                                          , R
                                                          , Vector
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

-- | Return a vector that's equal to the sum of the rows of a matrix.
sumRows :: Matrix R -> Vector R
sumRows = Matrix.fromList . map Matrix.sumElements . Matrix.toRows

-- | Are all the elements of matrices @a@ and @b@ within the given ϵ?
--
-- Will raise an exception if the matrices have incompatible
-- dimensions.
allWithin :: (Ord a, Matrix.Container c a, Num (c a))
          => a
          -- ^ ϵ, ∀i∀j. |aᵢⱼ - bᵢⱼ| < ϵ
          -> c a
          -> c a
          -> Bool
allWithin ϵ a b = Matrix.maxElement (abs (a - b)) < ϵ
