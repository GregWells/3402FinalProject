![](images/UTA-DataScience-Logo.png)

# Drone Delivery 

* This repository holds an attempt to apply model-free Reinforcement Learning techniques to optimize product delivery times as detailed in the "Drone Delivery" Kaggle challenge (https://www.kaggle.com/competitions/hashcode-drone-delivery/overview). 

## Overview

  * **Drone Delivery challenge:**  The task, as defined by the Kaggle challenge, is to develop code that will read a variable input file of system setup parameters and customer orders, and subsequently generate a submission file detailing each action that will be performed. The submission file will subsequently be scored by Kaggle based on how rapidly each order was delivered.   
  * **Solution method:** The approach in this repository leverages reinforcement learning (RL) to determine an efficient path to complete the orders. The problem was identified as fitting an **infinite-horizon, discounted return** equation:<br >
  ![](images/infiniteHorizonDiscountedReturn2.PNG)

This can be read as simply: The total reward equals the sum of the reward for each time step multiplied by the discount factor.<br>
<br/>
The &gamma;<sup>t</sup> factor in this equation is the discount factor. If &gamma;<sup>t</sup> is equal to zero, future rewards have no value, if &gamma;<sup>t</sup> is 1 then future rewards have no discount applied. Typical &gamma;<sup>t</sup> factors are .9-.99.
This scenario aligns well with model-free deep RL algorithms.  I compared the performance of 2 different RL algorithms, Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO).
  
  * **Summary of the performance achieved:** Ex: Our best model was able to predict the next day stock price within 23%, 90% of the time. At the time of writing, the best performance on Kaggle of this metric is 18%.

## Summary of Workdone

Include only the sections that are relevant and appropriate.

### Data

* Input:

   * Configuration file: CSV file containing challenge metrics (all data are integers):
     * Size of delivery area
     * Number of drones
     * Number of warehouses
     * Number of product types and weight for each type
     * Inventory of each warehouse
     * Order list detailing customer location, product numbers and quantities
 
   * Size: Training set consists of 30 drones, 10 warehouses, 400 product types, 1250 orders for 9396 total 
items.
* Code submission: 
   * This Kaggle challenge requires a code (notebook) submission which will process the test data set for subsequent scoring.
* Output:
   * Submission file requires a specific format for each action in a space-separated ASCII file, one action per line.

   

#### Preprocessing / Clean up

* No cleaning of the data was necessary for the training data set. All values were positive (or zero) integers. All orders were for corresponding valid product numbers. 

#### Data Visualization

While the training data set may by visualized in a number of different ways, the test data may not resemble this dataset at all. Designing a solution based on the training data distribution may lead to extreme overfitting. 


![Training Dataset:Warehouse and order locations](images/warehouseVsOrderLocations.PNG) 


Of note in the training dataset is that just over half of the orders can be fulfilled by utilizing the inventory of the closest warehouse. 

![Training Dataset:Warehouse proximity to orders](images/proximity.PNG) 



![Training Dataset:Product weight histogram](images/ProductWeightDistribution.PNG)

The maximum drone load for the training data is 200. 


### Problem Formulation

* As defined in the Kaggle challenge specification, the configuration parameters fall in the following ranges:
    * number of rows in the area of the simulation (1 ≤ n ≤ 10,000)
    * number of columns in the area of the simulation  (1 ≤ n ≤ 10,000)
    * number of drones available (1 ≤ D ≤ 1,000)
    * number of warehouses  (1 ≤ W ≤ 10,000)
    * number of products available (1 ≤ P ≤ 10,000)
    * number of customer orders  (1 ≤ C ≤ 10,000)
    * number of items per customer order  (1 ≤ I ≤ 10,000)
    * deadline of the simulation 1 (1 ≤ deadline ≤ 1,000,000)
    * maximum load of a drone (1 ≤ max load ≤ 10,000)

  * Required output file configuration (filename: submission.csv):
    * Line 0: Number of output lines in the output file following this quantity
    * Lines 1-n: Space-separated action lines e.g. '0 L 3 4 5', one set per line

  * Models
    * The structure of the challenge lends itself to a reinforcement learning approach and within that domain, a multi-discrete action and observation space as each decision (drone, warehouse, product, order) is discrete not continuous. This limits the available options within the OpenAI derived family:


![Stable Baselines3 reinforcement learning algorithms](images/RL_Algos.jpg)

 Source:[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html).
 

  * Loss, Optimizer, other Hyperparameters.
     
     * action parameters utilized:
         * Drone number
         * Load/Unload/Deliver 
         * Location (all locations were transformed from row, col to an unique integer value) 
         * Order number
     * observation parameters
         * sequence number: the AI needs information on how the timing affects choices
         * number of filled orders: also a timing input
         * drone location: drone location after the most recent action
         * drone payload: drone payload weight after the most recent action
         * previous action: (Load/Unload/Deliver)
         * most recent action: (Load/Unload/Deliver) This is probably unneccessary 
     * penalties(negative rewards) utilized:
         * Attempt to load/unload at a location other than a warehouse
         * Attempt to deliver a product for an order that has already been completed
         * Attempt to deliver a product for an order that is already "in flight"
         * Attempt to deliver a product that is not onboard the specified drone
         * Attempt to unload a product at the same warehouse that it originated
         * Attempt to load a product that is not in stock 
         * Attempt to load a drone beyond its capacity
     * rewards utilized:
         * Deliver an order to at the customer location
         * Load a product at a warehouse
         * Load a subsequent product at a warehouse (Bonus for multiple loads)
         * Unload a product at a warehouse
         * Unload a subsequent product at a warehouse (Bonus for multiple unloads)
         * Unload a product at a warehouse that has zero inventory of said product (Bonus)       

### Training

* Training sesison were typically several hours long even on a severely scaled back dataset of 55 orders.  
    * Software environment
        * Windows 10
        * Anaconda Navigator 2.3.2 
        * pyTorch version 1.13.0
        * gym 0.21.0
        * Stable Baselines 3
        * Algorithms: PPO & TRPO

    * Hardware:
       * CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz 
       * GPU: NVIDIA GeForce RTX 2080Ti  (Note: The frames-per-second were decreased by utilizing the GPU. Therefore, it was disabled for training runs) 

  
  * Training curves (loss vs epoch for test/train).
  ![Reduced dataset training curve](images/Tensorboard_PPOvsTRPO.PNG) 

      * Pink is a 6 hour PPO learning session with 55 orders/1 drone/10 warehouses
      * Gray is a 6 hour CPTO learning session with 55 orders/1 drone/10 warehouses
      
  ![Full-size Simulation training curve](images/FullSimTRPO.PNG)     
    
      * Blue is a 10 hour TRPO learning session with 9300 orders/30 drones/10 warehouses (full simulation)

  * Training was terminated when the mean score per episode was positive and flatlined for a significant period or trended negatively. 

#### Difficulties/challenges
  I had numerous difficulties during training. My initial goal was to attempt to coax intermodal transfers (warehouse-to-warehouse) to occur. In my non-ML-coded solutions, the score could be increased significantly by dedicating 30% of the drones to solely perform intermodal transfers.  Providing a positive reward for delivering an out-of-stock item to a warehouse, led to the AI finding this scoring opportunity and maximizing it by delivering a single out of stock item to a warehouse, then immediately reloading that product onto the drone and delivering it again:  rinse, wash, repeat...
  There were numerous "learning opportunities" like this and it required many iterations to configure a reward system to elicit the desired delivery (scoring) behavior. Simplifying the action/observation space would likely lead to a more simple ruleset. 

### Performance Comparison

* Key performance metrics:
    * Mean length of an episode - If an episode duration is configured to be of significant length to reach the final objective, when the AI reaches begins to reach the objective before the timer expires, this can be identified by the mean episode length decreasing.
    * Mean reward - If the reward configurations are tuned properly, the reward should dramatically increase as the actions transition from purely random toward more focused on positive rewards.
The performance in frames-per-second to train on the full dataset (9300 orders/30 drones/10 warehouses) is less than one-twentieth the rate of a reduced dataset of (55 orders/1 drone/10 warehouses). 

![](images/FullSimFPS.PNG)



### Conclusions

* The ability to closely tune the model action space to the problem is a key factor. For this challenge my utilization of the Stable Baselines 3 framework utilizing the multidiscrete action space was somewhat problematic. Spending significant time investigating configuration capabilities of the available frameworks before commencing any coding is key to assuring the action space is confined to the smallest region possible. As with any "traveling saleman" optimization, limiting the domain of possible choices to exclude unreasonable options is key. Tuning the penalties for invalid selections is challenging; it would be preferable to exclude the options from the action space altogether in advance - __reduce the dimensionality wherever practical__.

### Future Work

* I remain enthused to be able to facilitate an environment and an associated penalty/reward system that can elicit emergent behavior from an AI. The OpenAI gym environment can help provide a visual window into the RL training/testing process so I plan to continue with more simplistic challenges to continue my reinforcement learning education before returning to more complex action spaces/observation spaces..

* Future expansion possibilities:
    * An incremental step forward using this codeset would be to tune the reward system until both orders and intermodal transfers occur. 
    * Alternately, a secondary AI could be introduced to solely perform intermodal transfers in cooperation with the existing agent.
    * Alter the action sequence to facilitate a much smaller action space. Limiting the size of the space of choices that a drone needs to select from could significantly improve performance. Possibly abstracting the drone number altogether would be beneficial with the observation space simply showing "__A__ drone at __A CERTAIN LOCATION__ has __EXCESS CAPACITY: n___ and has these order opportunities for delivery."

## How to reproduce results

* To reproduce these results on a local PC:
    * download the busy_day.in file from the Kaggle challenge site: https://www.kaggle.com/competitions/hashcode-drone-delivery/data
    * Install Anaconda Navigator: https://docs.anaconda.com/navigator/install/
    * From the Anaconda navigator primary window, open a command window and install all neccessary modules
        * pip install numpy
        * pip install pandas
        * pip install matplotlib
        * pip install gym
        * pip install tensorboard
        * pip install sb3_contrib
        
     * From the Anaconda navigator primary window, open jupyter notebook
         * open the Drone_vXXX.ipynb
         * run each cell in succession
         * before executing the drone learn section (after is OK too):
             * To Monitor the learning:
                * from an Anaconda command window:
                * change dirctory to the directory the ipynb notebook was located in. 
                    e.g. cd C:\Users\greg\Documents\GitHub\DATA3402\Exams\Final\Drone\logs
                * tensorboard --logdir=. 
                * then open a browser window to:http://localhost:6006/
          * run the learning cell (training)
          * after a few minutes, refresh the tensorboard browser window, the stats will show (can be slow) refresh as needed
          
          * When results have plateaued, ep_rew_mean(episode rewards mean) has climbed sharply then eventually flattened out
          * stop the cell from running 
          * check the directory where tensorboard was started from. New subdirectories, logs and models, should exist there now.
          * navigate to the most recent model directory
          * copy the run number (10 digit number) and the highest zipfile # into the notated locations in DroneProcess_v.xxx.ipynb
          * run the DroneProcess 
          * validate the submission.csv file was generated

#### Key coding constructs to facilitate training other multidiscrete RL tasks: 

* Define the class:

>class DroneEnv(gym.Env): <br>
>    def __init__(self): <br>
>        super(DroneEnv, self).__init__() <br>
>        #define/initialize all variables  <br>
>        self.action_space = gym.spaces.MultiDiscrete([{action 0 size(positive int)},{action 1 size(positive int)},....]) <br>
>        self.observation_space =gym.spaces.MultiDiscrete([{obs 0 size(positive int)},{obs 1 size(positive int)},....]) <br>
> <br>
>    def step(self, action):     #This is the repetitive loop to perform based on the action the AI "guesses" <br>
>        self.done=False <br>
>        # Your code here to determine the amount of reward for the guess.... <br>
>        self.reward=.... <br>
>        self.info={}   #placeholder, not currently used <br>
>        if the objective has been met then set self.done=True <br>
>        self.observation= [{variables listed here MUST match the size as specified in the init section}]  <br>
>        self.observation = np.array(self.observation) <br>
>        return self.observation, self.reward, self.done, self.info <br>
> <br>
>    def reset(self): <br>
>        #set all variables to initial state <br>
>        #return the initial state (observation) to start the episode with <br>
>        self.observation= [{variables listed here MUST match the size as specified in the init section}]  <br>
>        self.observation = np.array(self.observation) <br>
>        return self.observation <br>

* Check the code:

>   #checkenv <br>
>   from stable_baselines3.common.env_checker import check_env <br>
>   env = DroneEnv()     #Change DroneEnv to the name of your class <br>
>   # This will check your environment and output warnings  <br>
>   check_env(env) <br>

*  Doublecheck
This will help find any mismatches between the code and the action and/or observation sizes. It randomly explores the action space to make sure the action and observation spaces do not exceed the specifications from the init statement. Getting an error here should be viewed positively. This (potentially) saves the effort of crashes midway through a training session. Increase the episode count times 10 or 100 if you feel confident. 

>   import stable_baselines3 <br>
>   stable_baselines3.common.env_checker import check_env <br>
>   rewardList=list() <br>
>   env = DroneEnv() <br>
>   episodes = 5 <br>
>   for episode in range(episodes): <br>
>       done = False <br>
>       obs = env.reset() <br>
>       while not done: <br>
>           random_action = env.action_space.sample() <br>
>           obs, reward, done, info = env.step(random_action) <br>
>           rewardList.append(reward) <br>

* Prepare to monitor training session:
To Monitor the learning:
>    from a command prompt: <br>
>    tensorboard --logdir={location of the training logs}  <br>
>    then open a browser window to:http://localhost:6006/   #6006 is the default port but can be changed <br>
    
*  Train:
>  import gym <br>
>  from stable_baselines3 import PPO <br>
>  from stable_baselines3.common.monitor import Monitor <br>
>  from stable_baselines3.common.evaluation import evaluate_policy <br>
>  import os <br>
>  #from droneenv import DroneEnv <br>

>  models_dir = f"models/{int(time.time())}/" <br>
>  logdir = f"logs/{int(time.time())}/" <br>

>  if not os.path.exists(models_dir): <br>
>      os.makedirs(models_dir) <br>

>  if not os.path.exists(logdir): <br>
>      os.makedirs(logdir) <br>

>  env = DroneEnv() <br>
>  env.reset() <br>
>  env = Monitor(env, logdir) <br>

>  model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir,device='cpu') <br>
>  TIMESTEPS = 10000 <br>
>  iters = 0 <br>
>  while True: <br>
>      iters += 1 <br>
>      model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO") <br>
>      model.save(f"{models_dir}/{TIMESTEPS*iters}") <br>
    

Refresh the tensorboard to watch training progress. Once the mean reward stabilizes, stop the training process.

* Run a simulation against the resultant model:

>  #Load a model and query <br>
>  # import libraries and packages <br>
>  import numpy as np <br>
>  import gym <br>
>  from stable_baselines3 import PPO <br>
>  import os <br>
>  from stable_baselines3.common.monitor import Monitor <br>
>  from stable_baselines3.common.evaluation import evaluate_policy <br>
>  #from droneenv import DroneEnv <br>

>  models_dir = f"models/1670818026/"                #Enter the dir of the model to utilize <br>

>  env = DroneEnv() <br>
>  env.reset() <br>
>  env = Monitor(env, logdir) <br>
>  model_path = f"{models_dir}/12750000.zip"    #Enter the model file to utilize <br>
>  model = PPO.load(model_path, env=env) <br>
>  episodes = 1 <br>
>   <br>
>  for ep in range(episodes): <br>
>      obs = env.reset() <br>
>      done = False <br>
>      i=0 <br>
>      while not done: <br>
        
>          action, _states = model.predict(obs) <br>
>          obs, reward, done, info = env.step(action) <br>
>          print("i:{} action: {}  reward:{}  obs:{} done:{}".format(i,action,reward,obs,done) ) <br>

* If the results meet expectations, this is complete. Otherwise, change the step() section and retrain.


### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.


### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations


> OpenAI Gym documentation: https://www.gymlibrary.dev/

> Stable Baselines 3 documentation: https://stable-baselines3.readthedocs.io/en/master/

> Nicholas Renotte YouTube channel: https://www.youtube.com/watch?v=Mut_u40Sqz4&t=4949s

> 'sentdex' YouTube channel: https://www.youtube.com/watch?v=yvwxbkKX9dc

> https://medium.com/intro-to-artificial-intelligence/key-concepts-in-reinforcement-learning-2af715dfbfa
