#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

#include <string>
#include <vector>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    bool auto_start = false;
    std::vector<char*> filtered_args;
    filtered_args.reserve(argc);
    filtered_args.push_back(argv[0]);
    for(int i = 1; i < argc; ++i)
    {
        if(std::string(argv[i]) == "--auto-start")
        {
            auto_start = true;
            continue;
        }
        filtered_args.push_back(argv[i]);
    }
    int filtered_argc = static_cast<int>(filtered_args.size());

    // Load parameters
    auto vm = param::helper(filtered_argc, filtered_args.data());

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + up] to enter FixStand mode.\n";
    std::cout << "And then press [R2 + A] to start controlling the robot.\n";

    if(auto_start)
    {
        spdlog::warn("Auto-start is enabled for simulation testing. Do not use this flag on a real robot.");
        sleep(2);
        fsm->force_state("FixStand");
        sleep(3);
        fsm->force_state("Velocity");
    }

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

