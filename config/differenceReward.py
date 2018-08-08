import datetime

def run(core):
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    print("Starting difference test at\n\t%s\n"%(dateTimeString))

    core.data["Number of Agents"] = 9
    core.data["Number of POIs"] = 4
    core.data["World Width"] = 30
    core.data["World Length"] = 30
    core.data["Coupling"] = 4
    core.data["Observation Radius"] = 4.0
    core.data["Minimum Distance"] = 1.0
    core.data["Steps"] = 50
    core.data["Trains per Episode"] = 100
    core.data["Tests per Episode"] = 1
    core.data["Number of Episodes"] = 2000
    
    perfSaveFileName = "log/difference/perf %s.csv"%(dateTimeString)
    trajSaveFileName = "log/difference/traj %s.csv"%(dateTimeString)
    
    # print the current Episode
    core.addTestBeginFunc(lambda data: print(data["Episode Index"], data["Global Reward"]))
    
    # NOTE: make sure functions are added to the list in the right order
    # Add Rover Domain Construction Functionality
    from code.world_setup import blueprintAgent, blueprintPoi, initWorld
    core.addTrainBeginFunc(blueprintAgent)
    core.addTrainBeginFunc(blueprintPoi)
    core.addWorldTrainBeginFunc(initWorld)
    core.addTestBeginFunc(blueprintAgent)
    core.addTestBeginFunc(blueprintPoi)
    core.addWorldTestBeginFunc(initWorld)
    
    
    # Add Rover Domain Dynamic Functionality
    from code.agent_domain import doAgentSense, doAgentProcess, doAgentMove
    core.addWorldTrainStepFunc(doAgentSense)
    core.addWorldTrainStepFunc(doAgentProcess)
    core.addWorldTrainStepFunc(doAgentMove)
    core.addWorldTestStepFunc(doAgentSense)
    core.addWorldTestStepFunc(doAgentProcess)
    core.addWorldTestStepFunc(doAgentMove)
    
    # Add Agent Position Trajectory History Functionality
    from code.trajectory_history import createTrajectoryHistories, updateTrajectoryHistories, saveTrajectoryHistories
    core.addWorldTrainBeginFunc(createTrajectoryHistories)
    core.addWorldTrainStepFunc(updateTrajectoryHistories)
    core.addTrialEndFunc(saveTrajectoryHistories(trajSaveFileName))
    core.addWorldTestBeginFunc(createTrajectoryHistories)
    core.addWorldTestStepFunc(updateTrajectoryHistories)

    
    
    # Add Agent Reward Functionality
    from code.reward import assignGlobalReward, assignDifferenceReward
    core.addWorldTrainEndFunc(assignDifferenceReward)
    core.addWorldTestEndFunc(assignGlobalReward)
    
    # Add Performance Recording Functionality
    from code.reward_history import printGlobalReward, saveRewardHistory, createRewardHistory, updateRewardHistory
    core.addTrialBeginFunc(createRewardHistory)
    core.addTestEndFunc(updateRewardHistory)
    core.addTrialEndFunc(saveRewardHistory(perfSaveFileName))
    
    # Add CCEA Functionality (all functionality below are dependent and are displayed together for easy accessibility)
    from code.ccea import initCcea, assignCceaPolicies, rewardCceaPolicies, evolveCceaPolicies, assignBestCceaPolicies
    core.addTrialBeginFunc(initCcea(input_shape= 8, num_outputs=2, num_units = 16))
    core.addWorldTrainBeginFunc(assignCceaPolicies)
    core.addWorldTrainEndFunc(rewardCceaPolicies)
    core.addTrainEndFunc(evolveCceaPolicies)
    core.addWorldTestBeginFunc(assignBestCceaPolicies)

    core.run()



"""
TODO
make sim_core code
make config code as just values to change
the problem with turning some codes into classes is that I can not easily change 
    the parameters mid trial
give Poi's a value (changes setup, reward and sense functions)
"""
