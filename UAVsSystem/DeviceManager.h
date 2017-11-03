#pragma once
#include <vector>
#include <string>
#include "thrust\device_ptr.h"

class DeviceManager{
private:
	static const unsigned int DEFAULT_DEVICE_ID;
	static std::vector<unsigned int> useCount;
	unsigned int deviceId;
private:
	std::string getClassName() const;

	void setDeviceId(const unsigned int &deviceId);
	void releaseDevice() const;

	void incrUseCount();
	unsigned int decrUseCountAndGet();

	virtual const thrust::device_ptr<void> getPtr() = 0;
protected:
	DeviceManager();
	DeviceManager(const unsigned int &deviceId);
	virtual ~DeviceManager() = 0;
	void switch2Device() const;

	std::string getDeviceIdStr() const;
public:
	unsigned int getDeviceId() const;
};

