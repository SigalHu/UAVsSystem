#include <cstdarg>
#include <memory>
#include "StringUtils.h"

std::string StringUtils::format(const std::string &_Format, ...){
	int final_n, n = ((int)_Format.size()) * 2;
	std::unique_ptr<char[]> formatted;
	va_list ap;
	while (true) {
		formatted.reset(new char[n]);
		strcpy(&formatted[0], _Format.c_str());
		va_start(ap, _Format);
		final_n = vsnprintf(&formatted[0], n, _Format.c_str(), ap);
		va_end(ap);
		if (final_n < 0 || final_n >= n)
			n += abs(final_n - n + 1);
		else
			break;
	}
	return std::string(formatted.get());
}
