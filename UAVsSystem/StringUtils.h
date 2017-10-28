#pragma once
#include <string>

class StringUtils final{
private:
	virtual ~StringUtils() = 0;
public:
	static std::string format(const std::string &_Format, ...);
};

