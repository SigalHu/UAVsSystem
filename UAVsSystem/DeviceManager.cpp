#include "DeviceManager.h"

const unsigned int DeviceManager::DEFAULT_DEVICE_ID = 0;
vector<unsigned int> DeviceManager::useCount(CudaUtils::getDeviceCount(), 0);

DeviceManager::DeviceManager(){
	setDeviceId(DEFAULT_DEVICE_ID);
	switch2Device();
	incrUseCount();
}

DeviceManager::DeviceManager(const unsigned int &deviceId){
	setDeviceId(deviceId);
	switch2Device();
	incrUseCount();
}

DeviceManager::~DeviceManager(){
	if (0 == decrUseCountAndGet()){
		releaseDevice();
	}
}

void DeviceManager::setDeviceId(const unsigned int &deviceId){
	if (deviceId >= DeviceManager::useCount.size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(deviceId),
		MacroUtils_VariableName(deviceId).append(" must be less than the number of GPU devices."));
	this->deviceId = deviceId;
}

unsigned int DeviceManager::getDeviceId() const{
	return this->deviceId;
}

void DeviceManager::releaseDevice() const{
	switch2Device();
	CudaUtils::resetDevice();
}

void DeviceManager::incrUseCount(){
	++DeviceManager::useCount[getDeviceId()];
}

unsigned int DeviceManager::decrUseCountAndGet(){
	return (--DeviceManager::useCount[getDeviceId()]);
}

void DeviceManager::switch2Device() const{
	CudaUtils::setDevice(getDeviceId());
}

string DeviceManager::getDeviceIdStr() const{
	return MacroUtils_VariableName(deviceId);
}