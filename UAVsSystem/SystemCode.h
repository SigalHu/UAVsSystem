#pragma once
#include <string>

class SystemCode final{
	friend class SystemCodeEnum;
private:
	const unsigned int code;
	const std::string info;
private:
	SystemCode() = default;
	SystemCode(const SystemCode&) = default;
	SystemCode(const unsigned int &code, const std::string &info);
public:
	~SystemCode();

	const unsigned int& getCode() const;
	const std::string& getInfo() const;
};

