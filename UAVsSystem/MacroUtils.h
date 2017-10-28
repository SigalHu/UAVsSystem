#pragma once

#define MacroUtils_VariableName(_Variable) std::string((void(_Variable),#_Variable))
#define MacroUtils_FunctionName(_Function) std::string((void(&_Function),#_Function))
#define MacroUtils_CurFunctionName() std::string(__FUNCTION__)
#define MacroUtils_ClassName(_Class) std::string(typeid(_Class).name())