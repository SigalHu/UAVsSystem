#pragma once

#include <string>
using namespace std;

class SystemCode final{
private:
	const unsigned int code;
	const string info;
public:
	static const SystemCode CUDA_RUNTIME_ERROR;
	static const SystemCode CUDA_CALL_ERROR;

	static const SystemCode NULL_POINTER;
	static const SystemCode OUT_OF_RANGE;
private:
	SystemCode() = default;
	SystemCode(const SystemCode&) = default;
	SystemCode(const unsigned int &code, const string &info);
public:
	~SystemCode();

	const unsigned int& getCode() const;

	const string& getInfo() const;
};

