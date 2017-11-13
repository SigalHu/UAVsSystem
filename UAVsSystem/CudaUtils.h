#pragma once
#include <string>

class CudaUtils{
private:
	static std::string getClassName();
public:
	virtual ~CudaUtils() = 0;
};