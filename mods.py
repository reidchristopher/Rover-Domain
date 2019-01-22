import datetime
from code.reward import calc_global_reward, calc_difference_reward, calc_dpp_reward # Agent Reward


class Mod:

    def global_reward_mod(data):
        data["Mod Name"] = "global"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_global_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

    def difference_reward_mod(data):
        data["Mod Name"] = "difference"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_difference_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

    def dpp_reward_mod(data):
        data["Mod Name"] = "dpp"

        date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
        print("Starting %s test at\n\t%s"%(data["Mod Name"], date_time_string))

        # Agent Reward
        data["Reward Function"] = calc_dpp_reward

        data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)

        data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
            (data["Specifics Name"], data["Mod Name"], date_time_string)



    # def globalRewardSizeCurrMod10(data):
    #     data["Schedule"] = ((10.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "globalSizeCurr10"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] =calc_global_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardSizeCurrMod20(data):
    #     data["Schedule"] = ((20.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "globalSizeCurr20"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #calc_dpp_reward
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] =calc_global_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #
    # def globalRewardSizeCurrMod30(data):
    #     data["Schedule"] = ((30.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "globalSizeCurr30"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] =calc_global_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardSizeCurrMod40(data):
    #     data["Schedule"] = ((40.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "globalSizeCurr40"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] =calc_global_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #
    #
    # def globalRewardCoupCurrMod1(data):
    #     data["Schedule"] = data["Schedule"] = ((1, 2000), (6, 3000))
    #     data["Mod Name"] = "globalCoupCurr1"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_global_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardCoupCurrMod2(data):
    #     data["Schedule"] = data["Schedule"] = ((2, 2000), (6, 3000))
    #     data["Mod Name"] = "globalCoupCurr2"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_global_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardCoupCurrMod3(data):
    #     data["Schedule"] = data["Schedule"] = ((3, 2000), (6, 3000))
    #     data["Mod Name"] = "globalCoupCurr3"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_global_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardCoupCurrMod4(data):
    #     data["Schedule"] = data["Schedule"] = ((4, 2000), (6, 3000))
    #     data["Mod Name"] = "globalCoupCurr4"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_global_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def globalRewardCoupCurrMod5(data):
    #     data["Schedule"] = data["Schedule"] = ((5, 2000), (6, 3000))
    #     data["Mod Name"] = "globalCoupCurr5"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_global_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # ##################################################################################
    #
    #
    #
    # def differenceRewardSizeCurrMod10(data):
    #     data["Schedule"] = ((10.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "differenceSizeCurr10"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardSizeCurrMod20(data):
    #     data["Schedule"] = ((20.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "differenceSizeCurr20"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #
    # def differenceRewardSizeCurrMod30(data):
    #     data["Schedule"] = ((30.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "differenceSizeCurr30"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardSizeCurrMod40(data):
    #     data["Schedule"] = ((40.0, 2000), (50.0,3000))
    #     data["Mod Name"] = "differenceSizeCurr40"
    #     trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    #     testBeginFuncCol.insert(0, restoreWorldSize)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #
    #
    # def differenceRewardCoupCurrMod1(data):
    #     data["Schedule"] = data["Schedule"] = ((1, 2000), (6, 3000))
    #     data["Mod Name"] = "differenceCoupCurr1"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardCoupCurrMod2(data):
    #     data["Schedule"] = data["Schedule"] = ((2, 2000), (6, 3000))
    #     data["Mod Name"] = "differenceCoupCurr2"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardCoupCurrMod3(data):
    #     data["Schedule"] = data["Schedule"] = ((3, 2000), (6, 3000))
    #     data["Mod Name"] = "differenceCoupCurr3"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardCoupCurrMod4(data):
    #     data["Schedule"] = data["Schedule"] = ((4, 2000), (6, 3000))
    #     data["Mod Name"] = "differenceCoupCurr4"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    # def differenceRewardCoupCurrMod5(data):
    #     data["Schedule"] = data["Schedule"] = ((5, 2000), (6, 3000))
    #     data["Mod Name"] = "differenceCoupCurr5"
    #     trainBeginFuncCol.insert(0, setCurriculumCoupling)
    #     testBeginFuncCol.insert(0, restoreCoupling)
    #
    #     date_time_string = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    #     print("Starting %s test at\n\t%s\n"%(data["Mod Name"], date_time_string))
    #
    #     # Agent Reward
    #     data["Reward Function"] = calc_difference_reward
    #
    #
    #     data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)
    #
    #     data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
    #         (data["Specifics Name"], data["Mod Name"], date_time_string)