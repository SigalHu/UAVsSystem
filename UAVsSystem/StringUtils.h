#pragma once

#include <cstdarg>
#include <string>
using namespace std;

class StringUtils final{
public:
	virtual ~StringUtils() = 0;

	static string format(const char *_Format, ...);
};

