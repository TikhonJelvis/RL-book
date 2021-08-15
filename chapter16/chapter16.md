## RL in Real-World Finance: Reality versus Hype, Present versus Future {#sec:concluding-chapter}

* Modeling operational details and business frictions - the most important thing
* Build a sampling model (simulator) estimated from real data, with human-augmentation (including scenarios, family of simulators), and run RL on this simulated environment
* In practice, one needs to build an entire ecosystem of: data management, software engineering, model development platform, model deployment platform, debuggability, measurements/instrumentation, testing, change management, iterative development, explainability, product management/user stories etc. The goal is to build a successful product, not just a successful model.
* imposing human views on top of a model/simulator
* blending model-based RL with model-free RL. 3 types of learning going on simultaneously. Design of such a real-time updating system
* simple models/closed form solutions first, then capturing frictions/constraints etc.
* in practice we rarely need to solve for optima. Close to optima good enough. Data quality issues make “optimization” meaningless.
* reasoning about uncertainties in data and uncertainties in model
* entire ecosystem not just models and algorithms
* feedback from production performance
* need for PMs
* need for scientists understanding business deeply and developing intuition connecting business to the mathematics Embeddings, Causality and RL: A deadly combination for the future
