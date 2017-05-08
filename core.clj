(ns multi-armed-bandits.core)

(def next-rand-gauss (atom nil)) ;; used in function rand-gauss

(defn rand-gauss [m s]
  "Generates random number from normal distribution N('m', 's'^2).
  Uses atom 'next-rand-gauss' to save computation."
  (if @next-rand-gauss
   (let [result (+ m (* s @next-rand-gauss))]
       (reset! next-rand-gauss nil)
       result)
   (let [x (rand)
         y (rand)
         r (-> x (Math/log) (* -2) (Math/sqrt))]
    (reset! next-rand-gauss (-> (* 2 Math/PI y) (Math/sin) (* r)))
    (-> (* 2 Math/PI y) (Math/cos) (* r s) (+ m)))))

(defn generate-bandits
  "Generates 'n' bandits.
  Each of them has variance = 1 and mean from normal distribution N('m', 's'^2).
  Returns sequence of pairs [mean variance=1]. Each pair describes one bandit."
  [m s n]
  (->> (fn [] [(rand-gauss m s) 1])
       (repeatedly n)))

(defn draw-each-bandit-once
  "Function used to perform initial probing of bandits.
  Draws each bandit once. Assumes we know the variances of the bandits.
  After initial drawing each bandit is estimated by mean equal to the drawn number and variance equal to the true variance (in case of gaussian likelihood this correspondes to gaussian prior with infinite variance)."
  [bandits drawing-function]
  (map (fn [[miu sigma]] [(drawing-function miu sigma) sigma]) bandits))

(defn gaussian-inference
  "Performs inference after 'reward' has been measured assuming prior distribution N('old-miu', 'old-std-dev'^2) and gaussian likelihood.
  Returns pair ['new-miu' 'new-std-dev'] describing posterior distribution."
  [[old-miu old-std-dev] reward likelihood-std-dev]
  (let
    [old-variance (* old-std-dev old-std-dev)
     likelihood-variance (* likelihood-std-dev likelihood-std-dev)
     new-miu (-> (* old-miu likelihood-variance)
                 (+ (* reward old-variance))
                 (/ (+ old-variance likelihood-variance)))
     
     new-std-dev (-> (* old-variance likelihood-variance)
                     (/ (+ old-variance likelihood-variance))
                     (Math/sqrt))]
    
    [new-miu new-std-dev]))

(defn choose-next-bandit
  "Returns index of the bandit with the highest value of (mean + 'factor' * standard-deviation).
   'bandits' - collection of pairs [mean standard-deviation]"
  ([factor bandits]
   (->> bandits
        (apply max-key (fn [[miu std-dev]] (+ miu (* std-dev factor))))
        (.indexOf bandits)))

  ([bandits]
    (choose-next-bandit 2 bandits)))

(defn simulate-one-step
  "Performs one step of the simulation, i.e. draws one bandit and returns updated bandits and updated total numbers of draws from each bandit."
  [bandits drawing-function inference-function choosing-function [observed-bandits numbers-of-draws]]
  (let
    [choosen-bandit-index (choosing-function observed-bandits)
     choosen-bandit (nth bandits choosen-bandit-index)
     reward (apply drawing-function choosen-bandit)
     updated-numbers (update numbers-of-draws choosen-bandit-index inc)
     updated-bandits (update observed-bandits choosen-bandit-index inference-function reward (second choosen-bandit))]
   [updated-bandits updated-numbers]))
 
(defn simulate
  "Main function.
  Generates 'number-of-bandits' gaussian bandits. Each of them has mean reward drawn from normal distribution N(0, 'bandits-distribution-s'^2) and variance = 1.
  Initially draws each bandit once, then makes 'steps' additional drawing.
  Returns vector consisting of:
   'bandits' - collection of pairs [mean variance] defining each bandit,
   'observed-bandits' - vector of pairs [mean variance-of-mean] describing our knowledge of badits based on drawn numbers,
   'numbers-of-draws' - vector of intigers - total numbers of draws from each bandit."
   [bandits-distribution-s number-of-bandits steps]
   (let
     [bandits (generate-bandits 0 bandits-distribution-s number-of-bandits)
      initial-observed (vec (draw-each-bandit-once bandits rand-gauss))
      initial-numbers (vec (repeat number-of-bandits 1))
      [observed-bandits numbers-of-draws] (as-> [initial-observed initial-numbers] $
                                                (iterate (partial simulate-one-step bandits rand-gauss gaussian-inference (partial choose-next-bandit 3)) $)
                                                (nth $ steps))]
      [bandits observed-bandits numbers-of-draws]))
