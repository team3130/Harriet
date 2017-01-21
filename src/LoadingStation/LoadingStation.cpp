#include <iostream>
#include <thread>
#include <chrono>
#include "networktables/NetworkTable.h"

int main()
{
	NetworkTable::SetClientMode();
	NetworkTable::SetTeam(3130);
	std::shared_ptr<NetworkTable> table = NetworkTable::GetTable("/Jetson");
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << table->GetString("Command", "Nothing") << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << table->GetString("Command", "Nothing") << std::endl;
	return 0;
}

