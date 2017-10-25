#pragma once

#define MacroUtils_VariableName(_Variable) string(#_Variable)
#define MacroUtils_FunctionName() string(__FUNCTION__)
#define MacroUtils_ClassName(_Class) string(typeid(_Class).name())