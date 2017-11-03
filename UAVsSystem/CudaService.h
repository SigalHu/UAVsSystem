#pragma once
#include <string>

class CudaService{
private:
	static const unsigned int DEFAULT_DEVICE_ID;

	const unsigned int deviceId;
private:
	static std::string getClassName();

	std::string getDeviceIdStr() const;
public:
	CudaService();
	CudaService(const unsigned int &deviceId);
	virtual ~CudaService();

	virtual unsigned int getDeviceId() const;

	virtual std::string toString() const;
};

