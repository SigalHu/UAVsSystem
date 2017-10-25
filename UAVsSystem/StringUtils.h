#pragma once

#include <cstdarg>
#include <memory>
#include <string>
using namespace std;

class StringUtils final{
private:
	virtual ~StringUtils() = 0;
public:
	static string format(const string &_Format, ...);
};

