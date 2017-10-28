#pragma once
#include <vector>
#include "cuda_utils.h"
#include "common.h"
using namespace std;

class DeviceManager{
private:
	static const unsigned int DEFAULT_DEVICE_ID;
	static vector<unsigned int> useCount;
	unsigned int deviceId;
private:
	void setDeviceId(const unsigned int &deviceId);
	void releaseDevice() const;
	void incrUseCount();
	unsigned int decrUseCountAndGet();
private:
	virtual void* getPtr() = 0;
protected:
	DeviceManager();
	DeviceManager(const unsigned int &deviceId);
	virtual ~DeviceManager() = 0;
	void switch2Device() const;

	string getDeviceIdStr() const;
public:
	unsigned int getDeviceId() const;
};

