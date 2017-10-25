#pragma once

#include <string>
using namespace std;

class SystemCode final{
	friend class SystemCodeEnum;
private:
	const unsigned int code;
	const string info;
private:
	SystemCode() = default;
	SystemCode(const SystemCode&) = default;
	SystemCode(const unsigned int &code, const string &info);
public:
	~SystemCode();

	const unsigned int& getCode() const;
	const string& getInfo() const;
};

